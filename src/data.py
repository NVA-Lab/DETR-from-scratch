"""
DETR 모델을 위한 데이터 로딩 및 전처리 모듈

이 모듈은 COCO 형식의 데이터셋을 로드하고 DETR 모델에 맞게 전처리합니다.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import DetrImageProcessor


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    DETR 학습을 위한 커스텀 COCO Detection 데이터셋 클래스
    
    torchvision의 CocoDetection을 상속받아 DETR 모델의 입력 형식에 맞게
    이미지와 어노테이션을 전처리합니다.
    
    Args:
        img_folder (str): 이미지가 저장된 폴더 경로
        imageprocessor (DetrImageProcessor): DETR 이미지 전처리기
        train (bool): 학습 모드 여부. True일 경우 데이터 증강 적용
    """
    
    def __init__(
        self, 
        img_folder: str, 
        imageprocessor: DetrImageProcessor, 
        train: bool = True
    ):
        # 학습/검증 모드에 따라 어노테이션 파일 선택
        ann_file = os.path.join(
            img_folder, 
            "custom_train.json" if train else "custom_val.json"
        )
        super(CocoDetection, self).__init__(img_folder, ann_file)
        
        self.imageprocessor = imageprocessor
        self.train = train
        
        # 학습 시에만 색상/블러 기반의 안전한 증강 적용 (박스 수정 불필요)
        self.augment = (
            T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
                T.RandomAutocontrast(p=0.2),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ])
            if train
            else None
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        인덱스에 해당하는 이미지와 타겟을 반환
        
        Args:
            idx (int): 데이터셋 인덱스
            
        Returns:
            Tuple[torch.Tensor, Dict]: (pixel_values, target)
                - pixel_values: 전처리된 이미지 텐서
                - target: DETR 형식의 타겟 딕셔너리
        """
        # PIL 이미지와 COCO 형식의 타겟 읽기
        img, target = super(CocoDetection, self).__getitem__(idx)

        # 학습 시 이미지 컬러/블러 증강 (기하 변환 없음 → 박스 수정 불필요)
        if self.augment is not None:
            img = self.augment(img)

        # DETR 형식으로 이미지와 타겟 전처리 (리사이징 + 정규화)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.imageprocessor(images=img, annotations=target, return_tensors="pt")
        
        # 배치 차원 제거
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    데이터로더를 위한 배치 collate 함수
    
    여러 샘플을 하나의 배치로 묶고, 이미지 크기가 다를 경우 패딩을 적용합니다.
    
    Args:
        batch (List[Tuple]): (pixel_values, target) 튜플의 리스트
        
    Returns:
        Dict[str, Any]: 배치 딕셔너리
            - pixel_values: 패딩된 이미지 텐서 [B, C, H, W]
            - pixel_mask: 패딩 마스크 [B, H, W]
            - labels: 타겟 딕셔너리 리스트
    """
    # 배치에서 이미지와 타겟 분리
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 이미지 크기가 다를 경우를 대비해 패딩 적용
    # imageprocessor를 외부에서 접근해야 하므로 함수 래퍼로 감싸는 것을 권장
    # 현재는 전역 변수로 가정
    encoding = batch[0][0].new_zeros(1)  # dummy for type
    
    # 배치 딕셔너리 생성
    batch_dict = {
        'pixel_values': torch.stack(pixel_values),
        'pixel_mask': None,  # imageprocessor.pad를 사용할 경우 자동 생성
        'labels': labels
    }
    
    return batch_dict


def create_collate_fn(imageprocessor: DetrImageProcessor):
    """
    imageprocessor를 클로저로 캡처하는 collate_fn 팩토리 함수
    
    Args:
        imageprocessor (DetrImageProcessor): DETR 이미지 전처리기
        
    Returns:
        callable: collate 함수
    """
    def collate_fn_with_processor(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
        """배치 collate 함수 (imageprocessor 포함)"""
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # imageprocessor를 사용해 패딩 적용
        encoding = imageprocessor.pad(pixel_values, return_tensors="pt")
        
        batch_dict = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
        
        return batch_dict
    
    return collate_fn_with_processor


def create_dataloaders(
    img_folder: str,
    imageprocessor: DetrImageProcessor,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle_train: bool = True,
    shuffle_val: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    학습 및 검증 데이터로더 생성
    
    Args:
        img_folder (str): 데이터셋 루트 폴더 (train/val 하위 폴더 포함)
        imageprocessor (DetrImageProcessor): DETR 이미지 전처리기
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩 워커 수
        shuffle_train (bool): 학습 데이터 셔플 여부
        shuffle_val (bool): 검증 데이터 셔플 여부
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_dataloader, val_dataloader)
    """
    # 데이터셋 생성
    train_dataset = CocoDetection(
        img_folder=f'{img_folder}/train',
        imageprocessor=imageprocessor,
        train=True
    )
    
    val_dataset = CocoDetection(
        img_folder=f'{img_folder}/val',
        imageprocessor=imageprocessor,
        train=False
    )
    
    # collate 함수 생성
    collate_fn = create_collate_fn(imageprocessor)
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader




