"""
Main training script for Deformable DETR.
"""
import os
import torch
import pytorch_lightning as pl
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
import autorootcwd

from src.dataset import CocoDetection
from src.models import DeformableDetr
from src.utils import visualize_predictions

# Enable Tensor Core/TF32 and cuDNN autotuner for better throughput
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


class DetrCollator:
    """Top-level collator to avoid Windows pickling issues with local functions."""
    def __init__(self, imageprocessor):
        self.imageprocessor = imageprocessor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.imageprocessor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels,
        }

def main():
    """Main function to run the training and inference."""
    # --- 1. Data Loading and Preprocessing ---
    img_folder = "data/merged_tomato"
    # Use slow processor for compatibility with pad(return_tensors="pt")
    imageprocessor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr", use_fast=False)

    train_dataset = CocoDetection(img_folder=os.path.join(img_folder, 'train'), imageprocessor=imageprocessor)
    val_dataset = CocoDetection(img_folder=os.path.join(img_folder, 'val'), imageprocessor=imageprocessor, train=False)
    
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    collate_fn = DetrCollator(imageprocessor)

    # DataLoader performance tuning
    num_workers = 8
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=2,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- 2. Training ---
    model = DeformableDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=len(id2label))
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",  # or "16-mixed" if bf16 not supported
        max_steps=300,
        gradient_clip_val=0.1,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        enable_progress_bar=False,
        val_check_interval=0.5,
        default_root_dir='lightning_logs/deformable_detr',
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # --- 3. Inference and Visualization ---
    run_inference(model, val_dataset, id2label)


def run_inference(model, val_dataset, id2label):
    """Runs inference on a random validation image and visualizes the result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image_id = val_dataset.ids[np.random.randint(0, len(val_dataset.ids))]
    image_info = val_dataset.coco.loadImgs(image_id)[0]
    image_path = os.path.join(val_dataset.root, image_info['file_name'])
    image = Image.open(image_path)
    
    # Get pixel values from dataset to ensure consistent preprocessing
    pixel_values, _ = val_dataset[val_dataset.ids.index(image_id)]
    pixel_values = pixel_values.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

    visualize_predictions(image, outputs, id2label, threshold=0.1)


if __name__ == '__main__':
    main()
