import torch
import torch.nn as nn
from functools import partial
from src.models.mobile_blocks import ConvBN, InvertedBottleNeck
from src.models.parameter_vector import FederatedModelMixin


class FOMOMNv2Baseline(FederatedModelMixin, nn.Module):
    def __init__(
            self,
            in_channels=3,
            device_num=0,
            random_seed=42,
            num_class=5,
            cfg=None,
            normalization="batch_norm",
    ):
        torch.random.default_generator.manual_seed(random_seed + device_num)
        super().__init__()
        self.normalization = normalization
        blocks = [
            ConvBN(in_channels, 32, 3, 2, normalization=normalization)
        ]
        in_c = 32
        # [t, c, n, s]
        '''
        assume the size of input data is cx96x96
        '''
        if cfg is None:
            self.cfg = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
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
                    expansion_source="out",
                    normalization=normalization,
                )
                in_c = setting[1]
                blocks.append(layer)

        # out 96x6x6
        blocks.append(
            ConvBN(in_c, 32, 3, 1, normalization=normalization)
        )
        blocks.append(nn.Conv2d(32, num_class, 3, 1, padding=1))
        self.blocks = nn.ModuleList(blocks)

        self.all_modules = list(self.blocks)
        self.conv_modules = list(self.blocks[:-1])
        self.fc_modules = [self.classifier]
        self.finalize_model_setup(validate_parameter_split=True)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x

    @property
    def convbn1(self):
        return self.blocks[0]

    @property
    def head(self):
        return self.blocks[-2]

    @property
    def classifier(self):
        return self.blocks[-1]

class FOMOMNv2Alpha035(FOMOMNv2Baseline):
    def __init__(
            self,
            in_channels=3,
            device_num=0,
            random_seed=42,
            num_class=5,
            normalization="batch_norm",
    ):
        cfg = [
            [1, 8, 1, 1],
            [6, 8, 2, 2],
            [6, 16, 3, 2],
            [6, 24, 4, 2],
            [6, 32, 3, 1],
        ]
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            cfg,
            normalization=normalization,
        )


class FOMOMNv2BaselineGroupNorm(FOMOMNv2Baseline):
    def __init__(self, in_channels=3, device_num=0, random_seed=42, num_class=5):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="group_norm",
        )


class FOMOMNv2BaselineLayerNorm(FOMOMNv2Baseline):
    def __init__(self, in_channels=3, device_num=0, random_seed=42, num_class=5):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="layer_norm",
        )


class FOMOMNv2Alpha035GroupNorm(FOMOMNv2Alpha035):
    def __init__(self, in_channels=3, device_num=0, random_seed=42, num_class=5):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="group_norm",
        )


class FOMOMNv2Alpha035LayerNorm(FOMOMNv2Alpha035):
    def __init__(self, in_channels=3, device_num=0, random_seed=42, num_class=5):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            normalization="layer_norm",
        )


class FOMOLoss(nn.Module):
    def __init__(self, input_size=96, downsample_rate=16, num_classes=6):
        super(FOMOLoss, self).__init__()
        self.input_size = input_size
        self.downsample_rate = downsample_rate
        self.num_classes = num_classes
        if self.num_classes == 6:
            cls_weight = torch.tensor(
                [1, 127, 140, 83, 224, 125], dtype=torch.float32
            )
        elif self.num_classes == 2:
            cls_weight = torch.tensor([1, 5], dtype=torch.float32)
        else:
            raise NotImplementedError
        self.ce = nn.CrossEntropyLoss(weight=cls_weight)

    def forward(self, y_pred, y_true):
        return self.ce(y_pred, y_true)


class FOMOMetrics(nn.Module):
    def __init__(self, connectivity=8, num_classes=2, min_area=1, dist_thr=1.0, ignore_bg=True):
        super(FOMOMetrics, self).__init__()
        if connectivity == 4:
            self.nbrs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        else:
            self.nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        self.num_classes = num_classes
        self.min_area = min_area
        self.dist_thr = dist_thr
        self.ignore_bg = ignore_bg

    def _components(self, mask):
        H, W = mask.shape
        visited = torch.zeros((H, W), dtype=torch.bool, device=mask.device)
        comps = []
        for y in range(H):
            for x in range(W):
                if mask[y, x] and not visited[y, x]:
                    stack = [(y, x)]
                    visited[y, x] = True
                    comp = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        for dy, dx in self.nbrs:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                comp.append((ny, nx))
                    if len(comp) >= self.min_area:
                        comps.append(comp)
        return comps

    def _centers(self, comps, device):
        centers = []
        for comp in comps:
            ys = torch.tensor([p[0] for p in comp], dtype=torch.float32, device=device)
            xs = torch.tensor([p[1] for p in comp], dtype=torch.float32, device=device)
            centers.append((xs.add(0.5).mean(), ys.add(0.5).mean()))
        return centers

    def _match(self, pred_centers, gt_centers, device):
        if len(pred_centers) == 0:
            tp = torch.tensor(0.0, device=device)
            fp = torch.tensor(0.0, device=device)
            fn = torch.tensor(float(len(gt_centers)), device=device)
            return tp, fp, fn
        if len(gt_centers) == 0:
            tp = torch.tensor(0.0, device=device)
            fp = torch.tensor(float(len(pred_centers)), device=device)
            fn = torch.tensor(0.0, device=device)
            return tp, fp, fn

        pred_xy = torch.stack([torch.stack([p[0], p[1]]) for p in pred_centers], dim=0)
        gt_xy = torch.stack([torch.stack([g[0], g[1]]) for g in gt_centers], dim=0)

        pred_used = torch.zeros((pred_xy.shape[0],), dtype=torch.bool, device=device)
        gt_used = torch.zeros((gt_xy.shape[0],), dtype=torch.bool, device=device)

        tp = torch.tensor(0.0, device=device)
        thr2 = torch.tensor(self.dist_thr * self.dist_thr, device=device, dtype=pred_xy.dtype)

        for gi in range(gt_xy.shape[0]):
            dxy = pred_xy - gt_xy[gi].unsqueeze(0)
            d2 = (dxy[:, 0] * dxy[:, 0]) + (dxy[:, 1] * dxy[:, 1])
            d2 = torch.where(pred_used, torch.full_like(d2, float("inf")), d2)
            best_d2, best_pi = torch.min(d2, dim=0)
            if torch.isfinite(best_d2) and best_d2 <= thr2:
                tp = tp + 1.0
                gt_used[gi] = True
                pred_used[best_pi] = True

        fn = (~gt_used).sum().to(torch.float32)
        fp = (~pred_used).sum().to(torch.float32)
        return tp, fp, fn

    def counts(self, y_pred, y_true):
        with torch.no_grad():
            if y_pred.dim() != 4:
                raise ValueError(f"y_pred must be (B,C,H,W), got {tuple(y_pred.shape)}")
            B, C, H, W = y_pred.shape
            C_use = min(C, self.num_classes)

            if y_true.dim() == 4:
                if y_true.shape[0] != B or y_true.shape[2] != H or y_true.shape[3] != W:
                    raise ValueError(f"y_true shape mismatch: {tuple(y_true.shape)} vs y_pred {tuple(y_pred.shape)}")
                y_true_is_onehot = True
            elif y_true.dim() == 3:
                if y_true.shape[0] != B or y_true.shape[1] != H or y_true.shape[2] != W:
                    raise ValueError(f"y_true shape mismatch: {tuple(y_true.shape)} vs y_pred {tuple(y_pred.shape)}")
                y_true_is_onehot = False
            else:
                raise ValueError(f"y_true must be (B,C,H,W) or (B,H,W), got {tuple(y_true.shape)}")

            prediction_device = y_pred.device
            if y_true.device != prediction_device:
                raise ValueError(f"y_true and y_pred must be on the same device, got {y_true.device} vs {prediction_device}")

            metric_device = torch.device("cpu")
            tp_c = torch.zeros(C_use, dtype=torch.float32, device=metric_device)
            fp_c = torch.zeros(C_use, dtype=torch.float32, device=metric_device)
            fn_c = torch.zeros(C_use, dtype=torch.float32, device=metric_device)

            pred_cls = torch.argmax(y_pred, dim=1).detach().to(metric_device)
            if y_true_is_onehot:
                gt_cls = torch.argmax(y_true, dim=1).detach().to(metric_device)
            else:
                gt_cls = y_true.detach().to(metric_device)

            for b in range(B):
                pred_b = pred_cls[b]
                gt_b = gt_cls[b]

                for c in range(C_use):
                    if self.ignore_bg and c == 0:
                        continue

                    pred_mask = (pred_b == c)
                    gt_mask = (gt_b == c)

                    pred_comps = self._components(pred_mask)
                    gt_comps = self._components(gt_mask)

                    pred_centers = self._centers(pred_comps, metric_device)
                    gt_centers = self._centers(gt_comps, metric_device)

                    tp, fp, fn = self._match(pred_centers, gt_centers, metric_device)
                    tp_c[c] = tp_c[c] + tp
                    fp_c[c] = fp_c[c] + fp
                    fn_c[c] = fn_c[c] + fn

            return tp_c, fp_c, fn_c

    def metrics_from_counts(self, tp_c, fp_c, fn_c):
        if not (tp_c.shape == fp_c.shape == fn_c.shape):
            raise ValueError("FOMO TP, FP, and FN counts must have identical shapes")
        eps = 1e-12
        precision_c = tp_c / (tp_c + fp_c + eps)
        recall_c = tp_c / (tp_c + fn_c + eps)
        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

        start = 1 if self.ignore_bg and tp_c.numel() > 1 else 0
        valid = torch.arange(start, tp_c.numel(), device=tp_c.device)
        zero = tp_c.new_zeros(())
        precision_macro = precision_c[valid].mean() if valid.numel() else zero
        recall_macro = recall_c[valid].mean() if valid.numel() else zero
        f1_macro = f1_c[valid].mean() if valid.numel() else zero
        return precision_macro, recall_macro, f1_macro

    def forward(self, y_pred, y_true):
        metrics = self.metrics_from_counts(*self.counts(y_pred, y_true))
        return tuple(metric.to(y_pred.device) for metric in metrics)


FOMOLossPerson = partial(FOMOLoss, num_classes=2)
FOMOLossVehicle = partial(FOMOLoss, num_classes=6)
FOMOLossVehicleBinary = partial(FOMOLoss, num_classes=2)

FOMOMetricsPerson = partial(FOMOMetrics, num_classes=2)
FOMOMetricsVehicle = partial(FOMOMetrics, num_classes=6)
FOMOMetricsVehicleBinary = partial(FOMOMetrics, num_classes=2)

if __name__ == '__main__':
    from src.data_getter.fomo_dataset_getter import FomoDataset
    import fiftyone as fo
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    import tqdm
    # model = FOMOMNv2Baseline()

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

    for experiment_time in range(2):
        if experiment_time == 0:
            model = FOMOMNv2Alpha035(num_class=2)
        else:
            model = FOMOMNv2Baseline(num_class=2)
        train_fo = fo.Dataset.from_dir(
            dataset_dir='../../data/coco-2017-fomo-vehicle-bak/train',
            dataset_type=fo.types.FiftyOneDataset,
        )
        val_fo = fo.Dataset.from_dir(
            dataset_dir='../../data/coco-2017-fomo-vehicle-bak/validation',
            dataset_type=fo.types.FiftyOneDataset,
        )
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        train_dataset = FomoDataset(train_fo, transform=transform)
        val_dataset = FomoDataset(val_fo, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        criterion = FOMOLossVehicleBinary()
        optimizer = Adam(model.parameters(), lr=0.005)
        metrics = FOMOMetricsVehicleBinary()

        epochs = 50

        model.to(device)
        for epoch in range(epochs):
            model.train()
            train_progress_bar = tqdm.tqdm(train_dataloader, desc=f"{epoch+1}/{epochs}", unit="batch")
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            running_f1_score = 0.0
            n = 0

            label_list = []
            for img, label in train_progress_bar:
                # label_list.append(label.detach().cpu().view(-1).numpy())
                # print(label[0])

                optimizer.zero_grad()
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                precision, recall, f1 = metrics(output, label)
                running_precision += precision
                running_recall += recall
                running_f1_score += f1
                n += 1

                train_progress_bar.set_postfix(loss=f"{running_loss/n}", precision=f"{running_precision/n}", recall=f"{running_recall/n}", f1_score=f"{running_f1_score/n}")

            # label_1d_array = label_list[0]
            # for array in label_list[1:]:
            #     label_1d_array = np.concatenate((label_1d_array, array))
            # vals, counts = np.unique(label_1d_array, return_counts=True)
            # print(vals, counts)

            model.eval()
            test_progress_bar = tqdm.tqdm(val_dataloader, desc=f"{epoch + 1}/{epochs}", unit="batch")
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            running_f1_score = 0.0
            n = 0
            with torch.no_grad():
                for img, label in test_progress_bar:
                    img = img.to(device)
                    label = label.to(device)
                    output = model(img)
                    loss = criterion(output, label)

                    running_loss += loss.item()

                    precision, recall, f1 = metrics(output, label)
                    running_precision += precision
                    running_recall += recall
                    running_f1_score += f1
                    n += 1

                    test_progress_bar.set_postfix(loss=f"{running_loss/n}", precision=f"{running_precision/n}", recall=f"{running_recall/n}", f1_score=f"{running_f1_score/n}")


