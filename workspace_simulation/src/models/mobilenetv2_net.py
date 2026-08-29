import torchvision
import torch
import torch.nn as nn
from src.models.mobile_blocks import ConvBN, InvertedBottleNeck
from src.models.parameter_vector import FederatedModelMixin

# mobilenet_v2_alpha035 = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_0.35_96")


class MobileNetV2Small(FederatedModelMixin, nn.Module):
    def __init__(
            self,
            in_channels=1,
            device_num=0,
            random_seed=42,
            num_class=47,
            normalization="batch_norm",
    ):
        # 28x28x1
        torch.random.default_generator.manual_seed(random_seed + device_num)
        super().__init__()
        self.normalization = normalization
        self.convbn1 = ConvBN(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            normalization=normalization,
        )
        self.ibn1 = InvertedBottleNeck(16, 24, 6, 1, normalization=normalization)
        self.ibn2 = InvertedBottleNeck(24, 24, 6, 1, normalization=normalization)
        self.ibn3 = InvertedBottleNeck(24, 32, 6, 1, normalization=normalization)
        self.ibn4 = InvertedBottleNeck(32, 32, 6, 1, normalization=normalization)
        self.ibn5 = InvertedBottleNeck(32, 32, 6, 1, normalization=normalization)
        self.ibn6 = InvertedBottleNeck(32, 64, 6, 2, normalization=normalization)
        self.ibn7 = InvertedBottleNeck(64, 64, 6, 1, normalization=normalization)
        self.ibn8 = InvertedBottleNeck(64, 96, 6, 1, normalization=normalization)
        self.ibn9 = InvertedBottleNeck(96, 96, 6, 1, normalization=normalization)
        self.ibn10 = InvertedBottleNeck(96, 160, 6, 2, normalization=normalization)
        self.ibn11 = InvertedBottleNeck(160, 160, 6, 1, normalization=normalization)
        self.ibn12 = InvertedBottleNeck(160, 320, 6, 1, normalization=normalization)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(320, num_class)

        self.blocks = [
            self.convbn1,
            self.ibn1,
            self.ibn2,
            self.ibn3,
            self.ibn4,
            self.ibn5,
            self.ibn6,
            self.ibn7,
            self.ibn8,
            self.ibn9,
            self.ibn10,
            self.ibn11,
            self.ibn12,
            self.avgpool,
            self.flatten,
            self.classifier
        ]

        self.all_modules = self.blocks
        self.conv_modules = self.blocks[:-1]
        self.fc_modules = [self.classifier]
        self.finalize_model_setup(validate_parameter_split=True)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        out = x
        return out


class MobileNetV2SmallGroupNorm(MobileNetV2Small):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="group_norm",
        )


class MobileNetV2SmallLayerNorm(MobileNetV2Small):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="layer_norm",
        )


class MobileNetV2Baseline(FederatedModelMixin, nn.Module):
    def __init__(
            self,
            in_channels=1,
            device_num=0,
            random_seed=42,
            num_class=47,
            cfg=None,
            normalization="batch_norm",
    ):
        torch.random.default_generator.manual_seed(random_seed + device_num)
        super().__init__()
        self.normalization = normalization
        blocks = [
            ConvBN(in_channels, 16, 3, 2, normalization=normalization)
        ]
        in_c = 16
        # [t, c, n, s]
        if cfg is None:
            self.cfg = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ]
        else:
            self.cfg = cfg
        for setting in self.cfg:
            for j in range(setting[2]):
                layer = InvertedBottleNeck(
                    in_c,
                    setting[1],
                    setting[0],
                    stride=setting[3] if j == 0 else 1,
                    normalization=normalization,
                )
                in_c = setting[1]
                blocks.append(layer)
        blocks.append(
            ConvBN(in_c, 1280, 1, normalization=normalization)
        )
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.Flatten(1))
        blocks.append(nn.Linear(1280, num_class))
        self.blocks = nn.ModuleList(blocks)

        self.all_modules = list(self.blocks)
        self.conv_modules = list(self.blocks[:-1])
        self.fc_modules = [self.classifier]
        self.finalize_model_setup(validate_parameter_split=True)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        out = x
        return out

    def __repr__(self):
        return repr(nn.Sequential(*self.blocks))

    @property
    def convbn1(self):
        return self.blocks[0]

    @property
    def classifier(self):
        return self.blocks[-1]


class MobileNetV2BaselineGroupNorm(MobileNetV2Baseline):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="group_norm",
        )


class MobileNetV2BaselineLayerNorm(MobileNetV2Baseline):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="layer_norm",
        )


class MobileNetV2Alpha035(MobileNetV2Baseline):
    def __init__(
            self,
            in_channels=1,
            device_num=0,
            random_seed=42,
            num_class=47,
            normalization="batch_norm",
    ):
        cfg = [
            [1, 8, 1, 1],
            [6, 8, 2, 2],
            [6, 16, 3, 2],
            [6, 24, 4, 2],
            [6, 32, 3, 1],
            [6, 56, 3, 2],
            [6, 112, 1, 1]
        ]
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            cfg,
            normalization=normalization,
        )
        #self.transfer_mnv2_weights(mobilenet_v2_alpha035)

    def transfer_mnv2_weights(self, src_model: nn.Module):
        if self.normalization != "batch_norm":
            raise ValueError(
                "MobileNetV2 pretrained BatchNorm weights can only be "
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
                dst_layers.append((name, m))

        print(f"src conv+bn: {len(src_layers)}, dst conv+bn: {len(dst_layers)}")

        transferred = 0
        for (name_s, m_s), (name_d, m_d) in zip(src_layers[5:], dst_layers[7:]):

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


class MobileNetV2Alpha035GroupNorm(MobileNetV2Alpha035):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="group_norm",
        )


class MobileNetV2Alpha035LayerNorm(MobileNetV2Alpha035):
    def __init__(self, in_channels=1, device_num=0, random_seed=42, num_class=47):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="layer_norm",
        )


if __name__ == '__main__':
    from torch import optim
    from torch.utils.data import DataLoader
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

    for size in [32, 64, 96, 128, 224]:
        print(size, "tt"*20)
        model = MobileNetV2Baseline(num_class=10, in_channels=3).to(device)
        train_datasets = torchvision.datasets.CIFAR10(
                '../../data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((size, size)),
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
                    torchvision.transforms.Resize((size, size)),
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
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        loss_fn = nn.CrossEntropyLoss()

        # os.makedirs("test/MNv2A035", exist_ok=True)
        model.to(device)
        print('begin to train')
        for epoch in range(1, 11):
            model.train()
            train_running_loss, train_running_acc, n = 0, 0, 0
            train_loop = tqdm(train_loader, desc=f'Train [{epoch}/200]', unit='batches')
            for images, labels in train_loop:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_running_acc += (pred == labels).float().mean().item()
                n += 1

                train_loop.set_postfix(loss=f"{train_running_loss/n:.3f}", acc=f"{train_running_acc/n:.3f}")
            #scheduler.step()

            model.eval()
            val_running_loss, val_running_acc, n = 0, 0, 0
            test_loop = tqdm(test_loader, desc=f'Val [{epoch}/200]', unit='batches')
            with torch.no_grad():
                for images, labels in test_loop:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    pred = outputs.argmax(dim=1)
                    val_running_loss += loss.item()
                    val_running_acc += (pred == labels).float().mean().item()
                    n += 1

                    test_loop.set_postfix(loss=f"{val_running_loss/n:.3f}", acc=f"{val_running_acc/n:.3f}")

        # torch.save({
        #     'epoch': epoch,
        #     'train_loss': train_running_loss / n,
        #     'train_acc': train_running_acc / n,
        #     'val_loss': val_running_loss / n,
        #     'val_acc': val_running_acc / n
        # }, f'test/MNv2A035/checkpoint_epoch_{epoch}.pth')



    #model = MobileNetV1Small(num_class=10, in_channels=3).to(device)
    # train_datasets = torchvision.datasets.CIFAR10(
    #     '../../data', train=True, download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             (0.4915, 0.4823, 0.4468),
    #             (0.2470, 0.2435, 0.2616)
    #         )
    #     ])
    # )
    #
    # test_datasets = torchvision.datasets.CIFAR10(
    #     '../../data', train=False, download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             (0.4915, 0.4823, 0.4468),
    #             (0.2470, 0.2435, 0.2616)
    #         )
    #     ])
    # )
    #
    # train_loader = DataLoader(dataset=train_datasets, batch_size=64, shuffle=True, num_workers=1)
    # test_loader = DataLoader(dataset=test_datasets, batch_size=64, shuffle=False, num_workers=1)
    # optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8)
    # # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # loss_fn = nn.CrossEntropyLoss()
    #
    # os.makedirs("test/MNv1S", exist_ok=True)
    # print(len(train_loader))
    # print('begin to train')
    # for epoch in range(1, 50):
    #     model.train()
    #     train_running_loss, train_running_acc, n = 0, 0, 0
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = loss_fn(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         train_running_loss += loss.item()
    #         pred = outputs.argmax(dim=1)
    #         train_running_acc += (pred == labels).float().mean().item()
    #         n += 1
    #     # scheduler.step()
    #     print(f'train loss={train_running_loss / n:.4f}, train acc={train_running_acc / n:.4f}')
    #     print('*****************************************')
    #
    #     model.eval()
    #     val_running_loss, val_running_acc, n = 0, 0, 0
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             loss = loss_fn(outputs, labels)
    #             pred = outputs.argmax(dim=1)
    #             val_running_loss += loss.item()
    #             val_running_acc += (pred == labels).float().mean().item()
    #             n += 1
    #     print(f'val loss={val_running_loss / n:.4f}, val acc={val_running_acc / n:.4f}')
    #     print('++++++++++++++++++++++++++++++++++++++++++')

        # torch.save({
        #     'epoch': epoch,
        #     'train_loss': train_running_loss / n,
        #     'train_acc': train_running_acc / n,
        #     'val_loss': val_running_loss / n,
        #     'val_acc': val_running_acc / n
        # }, f'test/MNv1S/checkpoint_epoch_{epoch}.pth')
