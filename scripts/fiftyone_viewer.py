"""
FiftyOne 데이터셋 뷰어 스크립트 (설정 파일 기반)
=====================================================

이 스크립트는 YAML 설정 파일에 정의된 정보를 바탕으로
COCO 또는 YOLO 형식의 객체 탐지 데이터셋을 FiftyOne으로 로드하여 시각화합니다.

[사용 방법]

1. `configs/fiftyone_viewer.yaml` 파일을 열어 시각화할 데이터셋 정보를 수정합니다.

2. 아래 명령어를 실행합니다.
   python src/fiftyone_viewer.py --config configs/fiftyone_viewer.yaml

[설정 파일 인자 설명]
format: 데이터셋 형식 ('coco' 또는 'yolo').
image_path: 이미지 파일이 있는 디렉토리 경로.
labels_path: 라벨 파일 경로 (COCO의 경우 .json 파일, YOLO의 경우 .txt 파일들이 있는 디렉토리).
classes: (YOLO 전용) 클래스 이름이 정의된 .names 파일 경로.
dataset_name: FiftyOne에 표시될 데이터셋의 이름.
"""
import fiftyone as fo
import argparse
import os
import yaml
from types import SimpleNamespace

def load_dataset(config):
    """
    제공된 설정을 기반으로 FiftyOne 데이터셋을 로드합니다.
    """
    
    # 데이터셋이 이미 존재하면 삭제하여 새로 로드합니다.
    if fo.dataset_exists(config.dataset_name):
        fo.delete_dataset(config.dataset_name)
        print(f"Deleted existing dataset: '{config.dataset_name}'")

    print(f"Loading dataset '{config.dataset_name}' from format '{config.format}'...")

    dataset_type = None
    kwargs = {}

    # -----------------------------------------------------------------
    # 포맷별 로드 설정
    # -----------------------------------------------------------------
    
    if config.format == 'coco':
        dataset_type = fo.types.COCODetectionDataset
        kwargs['data_path'] = config.image_path
        kwargs['labels_path'] = config.labels_path # COCO는 labels.json 파일 경로
        
    elif config.format == 'yolo':
        dataset_type = fo.types.YOLOv5Dataset
        kwargs['data_path'] = config.image_path   # 이미지 디렉토리
        kwargs['labels_path'] = config.labels_path # 라벨(.txt) 디렉토리

        # YOLO는 별도의 클래스 이름 파일이 필요할 수 있습니다.
        if hasattr(config, 'classes') and config.classes:
            if os.path.exists(config.classes):
                with open(config.classes, 'r') as f:
                    classes_list = [line.strip() for line in f.readlines()]
                kwargs['classes'] = classes_list
                print(f"Loaded {len(classes_list)} classes from '{config.classes}'.")
            else:
                print(f"Warning: Classes file not found at '{config.classes}'.")
        else:
            print("Warning: YOLO format selected but no --classes file provided. "
                  "Detections will use integer labels.")

    else:
        print(f"Error: Unsupported format '{config.format}'.")
        print("Supported formats: 'coco', 'yolo'")
        return

    # -----------------------------------------------------------------
    # 데이터셋 로드
    # -----------------------------------------------------------------
    
    try:
        dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            name=config.dataset_name,
            **kwargs
        )
        
        # 데이터셋을 영구적으로 저장 (선택 사항)
        dataset.persistent = True

        print(f"\nSuccessfully loaded {len(dataset)} samples.")
        print(f"Dataset '{config.dataset_name}' is ready.")
        
        return dataset

    except Exception as e:
        print(f"\n--- 데이터 로드 중 오류 발생 ---")
        print(e)
        print("----------------------------------\n")
        print("경로와 형식이 올바른지 확인하세요.")
        if config.format == 'coco':
            print("> COCO 형식은 다음을 예상합니다:")
            print(f"  image_path: {config.image_path} (이미지 폴더)")
            print(f"  labels_path: {config.labels_path} (annotations.json 파일)")
        elif config.format == 'yolo':
            print("> YOLO 형식은 다음을 예상합니다:")
            print(f"  image_path: {config.image_path} (이미지 폴더)")
            print(f"  labels_path: {config.labels_path} (라벨 .txt 폴더)")
            if hasattr(config, 'classes'):
                print(f"  classes: {config.classes} (.names 파일)")
        return None

def main():
    parser = argparse.ArgumentParser(description="Load Object Detection datasets into FiftyOne Viewer using a config file.")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    # Load config from YAML file
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        # Convert dict to namespace object for dot notation access (e.g., config.format)
        config = SimpleNamespace(**config_dict)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
        return
    except Exception as e:
        print(f"Error parsing YAML file: {e}")
        return
    
    dataset = load_dataset(config)
    
    if dataset:
        # FiftyOne 앱 실행
        print("Launching FiftyOne App...")
        session = fo.launch_app(dataset)

        if session:
            print("FiftyOne App is running. The session will be kept alive.")
            print("Press Ctrl+C in this terminal to exit.")
            session.wait()
            print("Session ended.")
        else:
            print("Failed to launch FiftyOne App session.")

if __name__ == "__main__":
    main()