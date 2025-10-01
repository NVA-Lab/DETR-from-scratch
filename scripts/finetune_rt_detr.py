"""
Main training script for RT-DETR.
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
from src.models import RTDetr
from src.utils import visualize_predictions

def main():
    """Main function to run the training and inference."""
    # --- 1. Data Loading and Preprocessing ---
    img_folder = "data/merged_tomato"
    imageprocessor = AutoImageProcessor.from_pretrained("PekingU/rtdetr-r50vd")

    train_dataset = CocoDetection(img_folder=os.path.join(img_folder, 'train'), imageprocessor=imageprocessor)
    val_dataset = CocoDetection(img_folder=os.path.join(img_folder, 'val'), imageprocessor=imageprocessor, train=False)
    
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = imageprocessor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

    # --- 2. Training ---
    model = RTDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=len(id2label))
    trainer = pl.Trainer(accelerator="gpu", max_steps=300, gradient_clip_val=0.1, default_root_dir='lightning_logs/rt_detr')
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
    
    pixel_values, _ = val_dataset[val_dataset.ids.index(image_id)]
    pixel_values = pixel_values.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

    visualize_predictions(image, outputs, id2label, threshold=0.1)


if __name__ == '__main__':
    main()


