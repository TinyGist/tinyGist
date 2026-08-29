import torch
import torch.nn as nn
from src.models.mobile_blocks import DepthwiseSeparableConv, build_normalization_2d
from src.models.parameter_vector import FederatedModelMixin


class MobileNetV1Small(FederatedModelMixin, nn.Module):
    def __init__(
            self,
            in_channels=1,
            device_num=10,
            random_seed=42,
            num_class=10,
            normalization="batch_norm",
    ):
        super().__init__()
        torch.random.default_generator.manual_seed(random_seed + device_num)
        # Input: 3x32x32 (CIFAR-10)
        self.conv1 = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            # bias = True
            bias=normalization == "none",
        )
        self.norm1 = build_normalization_2d(normalization, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dsconv2 = DepthwiseSeparableConv(32, 64, normalization=normalization)
        self.dsconv3 = DepthwiseSeparableConv(64, 128, normalization=normalization)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16x16
        self.dsconv4 = DepthwiseSeparableConv(128, 128, normalization=normalization)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8
        self.dsconv5 = DepthwiseSeparableConv(128, 256, normalization=normalization)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # 1x1 output

        self.flat = nn.Flatten(1)
        self.classifier = nn.Linear(256, num_class)

        self.blocks = [
            self.conv1,
            self.norm1,
            self.relu1,
            self.dsconv2,
            self.dsconv3,
            self.pool1,
            self.dsconv4,
            self.pool2,
            self.dsconv5,
            self.pool3,
            self.flat,
            self.classifier
        ]

        self.all_modules = self.blocks
        self.conv_modules = [self.conv1, self.norm1, self.dsconv2, self.dsconv3, self.dsconv4, self.dsconv5]
        self.fc_modules = [self.classifier]
        self.finalize_model_setup(validate_parameter_split=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.pool1(x)
        x = self.dsconv4(x)
        x = self.pool2(x)
        x = self.dsconv5(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.classifier(x)
        return x


class MobileNetV1SmallNoBN(MobileNetV1Small):
    """MobileNetV1Small with convolution biases and no BatchNorm modules."""

    def __init__(self, in_channels=1, device_num=10, random_seed=42, num_class=10):
        super().__init__(
            in_channels=in_channels,
            device_num=device_num,
            random_seed=random_seed,
            num_class=num_class,
            normalization="none",
        )


class MobileNetV1SmallGroupNorm(MobileNetV1Small):
    """MobileNetV1Small with eight-group GroupNorm after every convolution."""

    def __init__(self, in_channels=1, device_num=10, random_seed=42, num_class=10):
        super().__init__(
            in_channels=in_channels,
            device_num=device_num,
            random_seed=random_seed,
            num_class=num_class,
            normalization="group_norm",
        )


class MobileNetV1SmallLayerNorm(MobileNetV1Small):
    """MobileNetV1Small with channel-wise LayerNorm after every convolution."""

    def __init__(self, in_channels=1, device_num=10, random_seed=42, num_class=10):
        super().__init__(
            in_channels=in_channels,
            device_num=device_num,
            random_seed=random_seed,
            num_class=num_class,
            normalization="layer_norm",
        )


if __name__ == '__main__':
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import DataLoader
    # import fiftyone as fo
    import torchvision
    import tqdm
    from torchvision.transforms import Normalize, Compose
    # from src.data_getter.kws_getter import SpeechCommandsDataset

    model = MobileNetV1Small(in_channels=3)
    model.to('cuda')
    lr = 1e-2
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    train_dataset = torchvision.datasets.CIFAR10(
        '../../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4915, 0.4823, 0.4468),
                (0.2470, 0.2435, 0.2616)
            )
        ])
    )

    test_dataset = torchvision.datasets.CIFAR10(
        '../../data', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4915, 0.4823, 0.4468),
                (0.2470, 0.2435, 0.2616)
            )
        ])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    EPOCHS = 10
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_pbar = tqdm.tqdm(train_dataloader, unit="batches", desc=f"Train Epoch {epoch}:")
        running_loss = 0.0
        running_accuracy = 0.0
        n = 0
        for data, label in train_pbar:
            data = data.to('cuda')
            label = label.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(output, dim=1)
            running_accuracy += (y_pred == label).float().mean().item()
            running_loss += loss.item()
            n += 1

            train_pbar.set_postfix(loss=f"{running_loss/n:.3f}", accuracy=f"{running_accuracy/n:.3f}")

        model.eval()
        test_pbar = tqdm.tqdm(test_dataloader, unit="batches", desc=f"Test Epoch {epoch}:")
        running_loss = 0.0
        running_accuracy = 0.0
        n = 0
        with torch.no_grad():
            for data, label in test_pbar:
                data = data.to('cuda')
                label = label.to('cuda')
                output = model(data)
                loss = criterion(output, label)
                y_pred = torch.argmax(output, dim=1)
                running_accuracy += (y_pred == label).float().mean().item()
                running_loss += loss.item()
                n += 1

                test_pbar.set_postfix(loss=f"{running_loss/n:.3f}", accuracy=f"{running_accuracy/n:.3f}")



