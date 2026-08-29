import torch
from torch import nn
from functools import partial
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.models.mobile_blocks import (
    ConvBN,
    SeparableConvBn,
    build_normalization_1d,
)
from src.models.parameter_vector import FederatedModelMixin


class MircoYOLO(FederatedModelMixin, nn.Module):
    def __init__(
            self,
            in_channels=3,
            device_num=0,
            random_seed=42,
            num_class=5,
            box_s=5,
            num_box=2,
            normalization="batch_norm",
    ):
        torch.random.default_generator.manual_seed(random_seed + device_num)
        super().__init__()
        self.normalization = normalization
        self.convbn1 = ConvBN(
            in_channels,
            64,
            4,
            2,
            0,
            normalization=normalization,
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.dwsc1 = SeparableConvBn(64, 128, 3, 1, 0, normalization=normalization)
        self.dwsc2 = SeparableConvBn(128, 128, 3, 1, 1, normalization=normalization)
        self.dwsc3 = SeparableConvBn(128, 128, 3, 1, 0, normalization=normalization)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dwsc4 = SeparableConvBn(128, 128, 3, 1, 1, normalization=normalization)
        self.dwsc5 = SeparableConvBn(128, 64, 3, 1, 0, normalization=normalization)
        self.dwsc6 = SeparableConvBn(64, 64, 3, 1, 1, normalization=normalization)
        self.dwsc7 = SeparableConvBn(64, 64, 3, 1, 0, normalization=normalization)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(
            1024,
            1024,
            bias=normalization == "none",
        )
        self.bn = build_normalization_1d(normalization, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(1024, box_s*box_s*num_class+num_box*5)
        
        self.blocks = [
            self.convbn1,
            self.maxpool1,
            self.dwsc1,
            self.dwsc2,
            self.dwsc3,
            self.maxpool2,
            self.dwsc4,
            self.dwsc5,
            self.dwsc6,
            self.dwsc7,
            self.maxpool3,
            self.flatten,
            self.head,
            self.bn,
            self.relu,
            self.classifier,
        ]

        self.all_modules = self.blocks
        self.conv_modules = [
            self.convbn1, self.dwsc1, self.dwsc2, self.dwsc3,
            self.dwsc4, self.dwsc5, self.dwsc6, self.dwsc7,
        ]
        self.fc_modules = [self.head, self.bn, self.classifier]
        self.finalize_model_setup(validate_parameter_split=True)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class MircoYOLOGroupNorm(MircoYOLO):
    def __init__(
            self,
            in_channels=3,
            device_num=0,
            random_seed=42,
            num_class=5,
            box_s=5,
            num_box=2,
    ):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            box_s,
            num_box,
            normalization="group_norm",
        )


class MircoYOLOLayerNorm(MircoYOLO):
    def __init__(
            self,
            in_channels=3,
            device_num=0,
            random_seed=42,
            num_class=5,
            box_s=5,
            num_box=2,
    ):
        super().__init__(
            in_channels,
            device_num,
            random_seed,
            num_class,
            box_s,
            num_box,
            normalization="layer_norm",
        )


class YoLoLoss(nn.Module):
    def __init__(self, lam_cor=5, lam_no_obj=0.5, num_boxes=2, num_grids=5, num_classes=5):
        super(YoLoLoss, self).__init__()
        self.lam_cor = lam_cor
        self.lam_no_obj = lam_no_obj
        self.num_boxes = num_boxes
        self.num_grids = num_grids
        self.num_classes = num_classes

        self.classification_end_idx = self.num_grids * self.num_grids * self.num_classes
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _pairwise_calculate_iou(xywh_t: torch.Tensor, xywh_p: torch.Tensor, eps=1e-7):
        p = xywh_p.unsqueeze(2)
        t = xywh_t.unsqueeze(1)

        t_x1 = t[..., 0]
        t_y1 = t[..., 1]
        t_x2 = t[..., 0] + t[..., 2]
        t_y2 = t[..., 1] + t[..., 3]
        p_x1 = p[..., 0]
        p_y1 = p[..., 1]
        p_x2 = p[...,0] + p[..., 2]
        p_y2 = p[..., 1] + p[..., 3]

        inter_x1 = torch.maximum(t_x1, p_x1)
        inter_y1 = torch.maximum(t_y1, p_y1)
        inter_x2 = torch.minimum(t_x2, p_x2)
        inter_y2 = torch.minimum(t_y2, p_y2)
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        t_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
        p_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)

        return inter_area / (t_area + p_area - inter_area + eps)

    def greedy_allocate_box(self, iou_p_t: torch.Tensor):
        iou_box_map = torch.zeros((iou_p_t.shape[0], iou_p_t.shape[1]), device=iou_p_t.device, dtype=torch.long)
        iou_p_t_flat = iou_p_t.view(-1, self.num_boxes*self.num_boxes)
        _, sorted_indices = torch.sort(iou_p_t_flat, descending=True, dim=-1)
        for i in range(sorted_indices.shape[0]):
            used_pred_idx = []
            used_true_idx = []
            for sorted_idx in sorted_indices[i]:
                pred_idx = (sorted_idx//self.num_boxes).item()
                true_idx = (sorted_idx%self.num_boxes).item()
                if pred_idx not in used_pred_idx and true_idx not in used_true_idx:
                    used_pred_idx.append(pred_idx)
                    used_true_idx.append(true_idx)
                    iou_box_map[i, pred_idx] = true_idx

        return iou_box_map

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        # first get target data slice
        y_pred = self.sigmoid(y_pred)
        cls_t = y_true[:, :self.classification_end_idx]
        cls_p = y_pred[:, :self.classification_end_idx]
        box_t = y_true[:, self.classification_end_idx:]
        box_p = y_pred[:, self.classification_end_idx:]

        cls_t = cls_t.view(-1, self.num_grids*self.num_grids, self.num_classes)
        cls_p = cls_p.view(-1, self.num_grids*self.num_grids, self.num_classes)
        box_t = box_t.view(-1, self.num_boxes, 5)
        box_p = box_p.view(-1, self.num_boxes, 5)

        # second match box using IoU
        iou_p_t = self._pairwise_calculate_iou(box_t[..., :4], box_p[..., :4]) # [Batch, num_box, num_box]
        iou_p_t_idx = self.greedy_allocate_box(iou_p_t)

        box_t = box_t.gather(1, iou_p_t_idx.unsqueeze(-1).expand(-1, -1, 5))
        # finally, begin to calculate loss
        obj_mask = box_t[..., 4] == 1
        no_obj_mask = box_t[..., 4] == 0
        loss_xy = ((box_t[..., :2] - box_p[..., :2])**2)[obj_mask].sum()
        loss_wh = ((box_t[..., 2:4]**0.5 - box_p[..., 2:4]**0.5)**2)[obj_mask].sum()
        loss_obj = ((1-box_p[..., 4])**2)[obj_mask].sum()
        loss_no_obj = ((0-box_p[..., 4])**2)[no_obj_mask].sum()
        cls_obj_mask = cls_t.any(dim=-1)
        loss_cls = ((cls_t - cls_p)**2)[cls_obj_mask].sum()

        loss = (loss_xy + loss_wh)*self.lam_cor + loss_obj + loss_no_obj*self.lam_no_obj + loss_cls
        loss = loss / y_pred.shape[0]
        return loss

class YoLoMAP(nn.Module):
    def __init__(self, num_classes, num_boxes=2, num_grids=5, iou_type="bbox"):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.num_boxes = num_boxes
        self.num_grids = num_grids
        self.num_classes = num_classes

        # y = [S*S*C + B*5]
        self.classification_end_idx = num_grids * num_grids * num_classes

        # torchmetrics mAP
        self.metric = MeanAveragePrecision(iou_type=iou_type)

    @staticmethod
    def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
        # xywh: [...,4] with x,y as top-left
        x, y, w, h = xywh.unbind(dim=-1)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def get_class_pred(self, cls: torch.Tensor, box: torch.Tensor):
        """
        cls: [B, S^2, C]
        box: [B, Box, 5] -> [x,y,w,h,conf]
        return:
          boxes_xyxy: [B, Box, 4]
          scores:     [B, Box]
          labels:     [B, Box]
        """
        S = self.num_grids
        C = self.num_classes
        eps = 1e-6

        center_x = box[..., 0] + box[..., 2] / 2
        center_y = box[..., 1] + box[..., 3] / 2

        # keep centers inside [0,1)
        center_x = center_x.clamp(0, 1 - eps)
        center_y = center_y.clamp(0, 1 - eps)

        center_row = torch.floor(center_y * S).long().clamp(0, S - 1)  # [B, Box]
        center_col = torch.floor(center_x * S).long().clamp(0, S - 1)  # [B, Box]
        center_idx = center_row * S + center_col                       # [B, Box]

        idx = center_idx.unsqueeze(-1).expand(-1, -1, C)               # [B, Box, C]
        cls_extract = cls.gather(dim=1, index=idx)                     # [B, Box, C]

        conf = box[..., 4]                                             # [B, Box]
        cls_score, cls_pred = cls_extract.max(dim=-1)                  # [B, Box], [B, Box]
        scores = conf * cls_score                                      # [B, Box]
        labels = cls_pred                                              # [B, Box]

        boxes_xyxy = self.xywh_to_xyxy(box[..., :4])                    # [B, Box, 4]
        return boxes_xyxy, scores, labels

    def get_class_gt(self, cls: torch.Tensor, box: torch.Tensor):
        """
        GT:
          cls: [B, S^2, C]
          box: [B, Box, 5] -> [x,y,w,h,obj] where obj in {0,1}
        return (per image lists):
          boxes_xyxy_list: list length B, each [M,4]
          labels_list:     list length B, each [M]
        """
        boxes_xyxy, _, labels = self.get_class_pred(cls, box)
        obj = box[..., 4] > 0.5  # [B, Box] True for GT boxes

        boxes_list = []
        labels_list = []
        for i in range(box.shape[0]):
            m = obj[i]
            boxes_list.append(boxes_xyxy[i][m])
            labels_list.append(labels[i][m].long())
        return boxes_list, labels_list

    @torch.no_grad()
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        During eval, call this on each batch to update metric.
        """
        y_pred = self.sigmoid(y_pred)

        cls_t = y_true[:, :self.classification_end_idx]
        cls_p = y_pred[:, :self.classification_end_idx]
        box_t = y_true[:, self.classification_end_idx:]
        box_p = y_pred[:, self.classification_end_idx:]

        cls_t = cls_t.view(-1, self.num_grids * self.num_grids, self.num_classes)
        cls_p = cls_p.view(-1, self.num_grids * self.num_grids, self.num_classes)
        box_t = box_t.view(-1, self.num_boxes, 5)
        box_p = box_p.view(-1, self.num_boxes, 5)

        # decode preds
        pred_boxes_xyxy, pred_scores, pred_labels = self.get_class_pred(cls_p, box_p)

        # decode gts (filter obj==1)
        gt_boxes_list, gt_labels_list = self.get_class_gt(cls_t, box_t)

        # build torchmetrics input: list[dict] length B
        preds = []
        targets = []
        B = box_p.shape[0]
        for i in range(B):
            preds.append({
                "boxes": pred_boxes_xyxy[i].detach(),
                "scores": pred_scores[i].detach(),
                "labels": pred_labels[i].long().detach(),
            })
            targets.append({
                "boxes": gt_boxes_list[i].detach(),
                "labels": gt_labels_list[i].detach(),
            })

        self.metric.update(preds, targets)

    def clear(self):
        self.metric.reset()

    def mAP_calculate(self):
        """
        Call once after all val/test batches have been forwarded.
        """
        return self.metric.compute()


YoLoLossPerson = partial(YoLoLoss, num_classes=1)
YoLoLossVehicle = partial(YoLoLoss, num_classes=5)
YoLoLossVehicleBinary = partial(YoLoLoss, num_classes=1)
YoLoMAPPerson = partial(YoLoMAP, num_classes=1)
YoLoMAPVehicle = partial(YoLoMAP, num_classes=5)
YoLoMAPVehicleBinary = partial(YoLoMAP, num_classes=1)


if __name__ == '__main__':
    model = MircoYOLO(num_class=5)
    from src.data_getter.coco_detection_getter import COCOGetter
    import fiftyone as fo
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torchvision.transforms import Compose, ToTensor, Normalize
    import tqdm


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

    train_dataset = fo.Dataset.from_dir(
        dataset_dir="../../data/coco-2017-yolo-vehicle/train",
        dataset_type=fo.types.COCODetectionDataset,
    )
    test_dataset = fo.Dataset.from_dir(
        dataset_dir="../../data/coco-2017-yolo-vehicle/validation",
        dataset_type=fo.types.COCODetectionDataset,
    )
    transform = Compose(
        [
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
         ]
    )

    train_dateset = COCOGetter(train_dataset, task="vehicle", transform=transform)
    test_dataset = COCOGetter(test_dataset, task="vehicle", transform=transform)

    train_dataloader = DataLoader(train_dateset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criteria = YoLoLossVehicle()
    mAP_calculator = YoLoMAPVehicle()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        global_loss = 0
        n = 0
        train_pbar = tqdm.tqdm(train_dataloader, desc=f"{epoch+1}/{epochs}", unit="batch")
        for images, targets in train_pbar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step()
            n += 1
            global_loss += loss.item()
            train_pbar.set_postfix(loss=f"{global_loss/n}")

        model.eval()
        global_loss = 0
        n = 0
        test_pbar = tqdm.tqdm(test_dataloader, desc=f"{epoch+1}/test", unit="batch")
        mAP_calculator.clear()
        with torch.no_grad():
            for images, targets in test_pbar:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criteria(outputs, targets)
                n += 1
                global_loss += loss.item()
                test_pbar.set_postfix(loss=f"{global_loss/n}")
                mAP_calculator(outputs, targets)
        mAP_value = mAP_calculator.mAP_calculate()
        tqdm.tqdm.write(f"TEST: epoch: {epoch}, metrics: {mAP_value}")
