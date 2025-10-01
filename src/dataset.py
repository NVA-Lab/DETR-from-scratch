import os
import torchvision

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Custom CocoDetection class to apply DETR-specific transforms.
    It is subclassed from torchvision's CocoDetection class.
    """
    def __init__(self, img_folder, imageprocessor, train=True):
        ann_file = os.path.join(img_folder, "train_annotations.json" if train else "val_annotations.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.imageprocessor = imageprocessor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.imageprocessor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target
