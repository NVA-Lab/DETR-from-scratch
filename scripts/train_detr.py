import argparse

import pytorch_lightning as pl
from transformers import DetrImageProcessor

from src.config import get_config, build_label_mappings_from_coco_dataset
from src.data import create_dataloaders
from src.model import Detr


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR on COCO-format dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="dataset root (contains train/val)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_backbone", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    # CLI 인자로 넘어온 값이 있을 경우 config를 덮어쓰기
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.lr is not None:
        cfg.lr = args.lr
    if args.lr_backbone is not None:
        cfg.lr_backbone = args.lr_backbone
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.gradient_clip_val is not None:
        cfg.gradient_clip_val = args.gradient_clip_val
    if args.accelerator is not None:
        cfg.accelerator = args.accelerator
    if args.seed is not None:
        cfg.seed = args.seed

    pl.seed_everything(cfg.seed, workers=True)

    imageprocessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_loader, val_loader = create_dataloaders(
        img_folder=cfg.data_dir,
        imageprocessor=imageprocessor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle_train=True,
        shuffle_val=False,
    )

    id2label, label2id = build_label_mappings_from_coco_dataset(train_loader.dataset)
    num_labels = len(id2label)

    model = Detr(
        num_labels=num_labels,
        lr=cfg.lr,
        lr_backbone=cfg.lr_backbone,
        weight_decay=cfg.weight_decay,
    )

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        max_steps=cfg.max_steps,
        gradient_clip_val=cfg.gradient_clip_val,
        default_root_dir=cfg.log_dir,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()


