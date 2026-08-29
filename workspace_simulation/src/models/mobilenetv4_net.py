import os
import torch
import torch.nn as nn
from src.models.mobile_blocks import ConvBN, UniversalInvertedBottleneck
from src.models.parameter_vector import FederatedModelMixin

# mobilenetv4_s=timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)


class MobileNetV4Small(FederatedModelMixin, nn.Module):
    def __init__(
            self,
            in_channels=1,
            device_num=0,
            random_seed=42,
            num_class=47,
            normalization="batch_norm",
    ):
        super().__init__()
        torch.random.default_generator.manual_seed(random_seed + device_num)
        self.num_class = num_class
        self.normalization = normalization
        # 1x28x28
        self.convbn1 = ConvBN(in_channels, 32, 3, 2, normalization=normalization)
        # 32x14x14
        self.convbn2 = ConvBN(32, 32, 3, 2, normalization=normalization)
        # 32x7x7
        self.convbn3 = ConvBN(32, 32, 1,1, normalization=normalization)
        self.convbn4 = ConvBN(32, 96, 3, 2, normalization=normalization)
        # 96x4x4
        self.convbn5 = ConvBN(96, 64, 1, 1, normalization=normalization)
        self.extra_dw1 = UniversalInvertedBottleneck(64, 96, 3.0, 5, 5, 2, se=True, normalization=normalization) # ExtraDW
        self.ib1 = UniversalInvertedBottleneck(96, 96, 2.0, 0, 3, 1, se=True, normalization=normalization) # IB
        self.ib2 = UniversalInvertedBottleneck(96, 96, 2.0, 0, 3, 1, se=True, normalization=normalization) # IB
        self.ib3 = UniversalInvertedBottleneck(96, 96, 2.0, 0, 3, 1, se=True, normalization=normalization) # IB
        self.ib4 = UniversalInvertedBottleneck(96, 96, 2.0, 0, 3, 1, se=True, normalization=normalization) # IB
        self.conv_next1 = UniversalInvertedBottleneck(96, 96, 4.0, 3, 0, 1, se=True, normalization=normalization) #ConvNext
        # out 96x2x2
        self.extra_dw2 = UniversalInvertedBottleneck(96, 128, 6.0, 3, 3, 2, se=True, normalization=normalization) #ExtraDW
        self.extra_dw3 = UniversalInvertedBottleneck(128, 128, 4.0, 5, 5, 1, se=True, normalization=normalization) #ExtraDW
        self.ib5 = UniversalInvertedBottleneck(128, 128, 4.0, 0, 5, 1, se=True, normalization=normalization)
        self.ib6 = UniversalInvertedBottleneck(128, 128, 3.0, 0, 5, 1, se=True, normalization=normalization)
        self.ib7 = UniversalInvertedBottleneck(128, 128, 4.0, 0, 3, 1, se=True, normalization=normalization)
        self.ib8 = UniversalInvertedBottleneck(128, 128, 4.0, 0, 3, 1, se=True, normalization=normalization)
        # out 128x1x1
        self.convbn6 = ConvBN(128, 960, 1, 1, normalization=normalization)
        # out 960x1x1
        self.gpooling = nn.AdaptiveAvgPool2d(1)
        # out 960x1x1
        self.convbn7 = ConvBN(960, 1280, 1, 1, normalization=normalization)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(1280, self.num_class)

        self.blocks = [
            self.convbn1,
            self.convbn2,
            self.convbn3,
            self.convbn4,
            self.convbn5,
            self.extra_dw1,
            self.ib1,
            self.ib2,
            self.ib3,
            self.ib4,
            self.conv_next1,
            self.extra_dw2,
            self.extra_dw3,
            self.ib5,
            self.ib6,
            self.ib7,
            self.ib8,
            self.convbn6,
            self.gpooling,
            self.convbn7,
            self.flatten,
            self.classifier
        ]

        self.all_modules = self.blocks
        self.conv_modules = self.blocks[:-1]
        self.fc_modules = [self.classifier]
        self.finalize_model_setup(validate_parameter_split=True)
        #self.transfer_mnv4_weights(mobilenetv4_s)

    def __repr__(self):
        return repr(nn.Sequential(*self.blocks))

    def forward(self, x):
        #out = self.sequential(x)
        for block in self.blocks:
            x = block(x)
        out = x
        return out

    def transfer_mnv4_weights(self, src_model: nn.Module):
        if self.normalization != "batch_norm":
            raise ValueError(
                "MobileNetV4 pretrained BatchNorm weights can only be "
                "transferred to the BatchNorm variant"
            )

        # collect all Conv2d + BN from the timm model
        src_layers = []
        for name, m in src_model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                src_layers.append((name, m))

        # collect Conv2d + BN from your model, but SKIP SEBlock convs
        dst_model = nn.Sequential(*self.blocks)
        dst_layers = []
        for name, m in dst_model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                # convs inside SEBlock live under "se_block"
                if "se_block" in name:
                    continue
                dst_layers.append((name, m))

        print(f"src conv+bn: {len(src_layers)}, dst conv+bn (no SE): {len(dst_layers)}")

        transferred = 0
        for (name_s, m_s), (name_d, m_d) in zip(src_layers, dst_layers):

            # ---- Conv2d ----
            if isinstance(m_s, nn.Conv2d) and isinstance(m_d, nn.Conv2d):
                # match out_channels + kernel size
                if (
                        m_s.weight.shape[0] == m_d.weight.shape[0]
                        and m_s.weight.shape[2:] == m_d.weight.shape[2:]
                ):
                    # special case: first conv, 3 -> 1 channels
                    if m_s.weight.shape[1] == 3 and m_d.weight.shape[1] == 1:
                        with torch.no_grad():
                            w = m_s.weight.data.mean(dim=1, keepdim=True)  # RGB → gray
                            m_d.weight.data.copy_(w)
                        transferred += 1
                        # print(f"conv (3->1): {name_s} -> {name_d}")
                    # normal case: same #input channels
                    elif m_s.weight.shape[1] == m_d.weight.shape[1]:
                        with torch.no_grad():
                            m_d.weight.data.copy_(m_s.weight.data)
                        transferred += 1
                        # print(f"conv: {name_s} -> {name_d}")

            # ---- BatchNorm2d ----
            elif isinstance(m_s, nn.BatchNorm2d) and isinstance(m_d, nn.BatchNorm2d):
                if m_s.num_features == m_d.num_features:
                    with torch.no_grad():
                        m_d.weight.data.copy_(m_s.weight.data)
                        m_d.bias.data.copy_(m_s.bias.data)
                        m_d.running_mean.data.copy_(m_s.running_mean.data)
                        m_d.running_var.data.copy_(m_s.running_var.data)
                    transferred += 1
                    # print(f"bn: {name_s} -> {name_d}")

        print(f"Transferred {transferred} conv/bn layers (backbone only).")


class MobileNetV4SmallGroupNorm(MobileNetV4Small):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="group_norm",
        )


class MobileNetV4SmallLayerNorm(MobileNetV4Small):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="layer_norm",
        )


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader
    from torch import optim
    from tqdm import tqdm


    def pick_gpu_with_most_free_mem():
        n = torch.cuda.device_count()
        assert n > 0, "Keine CUDA-GPU gefunden."
        free = []
        for i in range(n):
            f, t = torch.cuda.mem_get_info(i)  # bytes frei, total
            free.append((i, f))
        best = max(free, key=lambda x: x[1])[0]
        return best


    if torch.cuda.is_available():
        gpu = pick_gpu_with_most_free_mem()
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = MobileNetV4Small(in_channels=3, num_class=10)
    params = model.get_parameter_vector("all")
    model.load_parameter_vector(params, "all")
    parameters_conv = model.get_parameter_vector("conv")
    parameters_fc = model.get_parameter_vector("fc")
    model.load_parameter_vector(parameters_fc, "fc")
    model.load_parameter_vector(parameters_conv, "conv")
    train_datasets = torchvision.datasets.CIFAR10(
            '../../data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4915, 0.4823, 0.4468),
                    (0.2470, 0.2435, 0.2616)
                )
            ])
        )

    test_datasets = torchvision.datasets.CIFAR10(
            '../../data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4915, 0.4823, 0.4468),
                    (0.2470, 0.2435, 0.2616)
                )
            ])
        )

    train_loader = DataLoader(dataset=train_datasets, batch_size=50, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_datasets, batch_size=100, shuffle=False, num_workers=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("test", exist_ok=True)
    print(len(train_loader))
    print('begin to train')
    model = model.to(device)
    for epoch in range(1, 200):
        model.train()
        running_loss, running_acc, n = 0, 0, 0
        train_loop = tqdm(train_loader, desc=f'{epoch}/200', unit='batches')
        for images, labels in train_loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            running_acc += (pred == labels).float().mean().item()
            n += 1

            train_loop.set_postfix(loss=f"{running_loss/n}", acc=f"{running_acc/n}")
        #scheduler.step()

        model.eval()
        running_loss, running_acc, n = 0, 0, 0
        test_loop = tqdm(test_loader, desc=f'{epoch}/200', unit='batches')
        with torch.no_grad():
            for images, labels in test_loop:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                pred = outputs.argmax(dim=1)
                running_loss += loss.item()
                running_acc += (pred == labels).float().mean().item()
                n += 1

                test_loop.set_postfix(loss=f"{running_loss/n}", acc=f"{running_acc/n}")

