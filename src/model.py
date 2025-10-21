from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Detr(pl.LightningModule):
    """
    PyTorch Lightning 모듈로 구현한 DETR 학습 래퍼.

    - HF의 DetrForObjectDetection을 내부에 보유
    - 학습/검증 공통 스텝과 mAP 측정 로직 포함
    """

    def __init__(
        self,
        num_labels: int,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        score_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self.map_metric = MeanAveragePrecision(
            box_format="cxcywh", iou_type="bbox", class_metrics=False
        )
        self.score_threshold = score_threshold

    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None
    ):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def _update_map(
        self, outputs: Any, labels: List[Dict[str, torch.Tensor]]
    ) -> None:
        """DETR 출력과 HF 라벨을 torchmetrics 형식으로 변환하여 mAP를 업데이트."""
        probs = outputs.logits.softmax(-1)[..., :-1]
        scores, pred_labels = probs.max(-1)
        pred_boxes = outputs.pred_boxes

        preds: List[Dict[str, torch.Tensor]] = []
        targets: List[Dict[str, torch.Tensor]] = []

        batch_size, num_queries, _ = pred_boxes.shape
        for i in range(batch_size):
            keep = scores[i] > self.hparams.score_threshold
            preds.append(
                {
                    "boxes": pred_boxes[i][keep].detach().cpu(),
                    "scores": scores[i][keep].detach().cpu(),
                    "labels": pred_labels[i][keep].detach().cpu(),
                }
            )

        for t in labels:
            targets.append(
                {
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["class_labels"].detach().cpu(),
                }
            )

        if len(preds) > 0:
            self.map_metric.update(preds, targets)

    def common_step(self, batch: Dict[str, Any], batch_idx: int):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask")
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item(), prog_bar=False)

        with torch.no_grad():
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
            outputs = self.model(
                pixel_values=batch["pixel_values"], pixel_mask=batch.get("pixel_mask")
            )
            self._update_map(outputs, labels)

        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.map_metric.compute()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log(f"val_{k}", v, prog_bar=True)
                continue
            if torch.is_tensor(v) and v.ndim == 0:
                self.log(f"val_{k}", v, prog_bar=True)
                continue
            if torch.is_tensor(v) and v.ndim > 0 and v.numel() > 0:
                if k.endswith("_per_class"):
                    self.log(f"val_{k}_mean", v.float().mean(), prog_bar=False)
                continue
        self.map_metric.reset()

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer


