import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.dataset_utils import check, separate_data, split_data, save_file
from torch.utils.data import DataLoader

# reproducibility
torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(1)
np.random.seed(1)

dir_path = "office/"
# 처리할 도메인 디렉토리 리스트
domain_names = ['amazon/', 'dslr/', 'webcam/']
num_classes = 31

def generate_office31_mixed(dir_path, num_clients, num_classes, niid, balance, partition):
    # 기본 디렉토리 및 서브디렉토리 생성
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")

    # 이미 생성된 데이터인지 확인
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        print("이미 데이터셋이 존재합니다. 생성 작업을 건너뜁니다.")
        return

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    all_images, all_labels = [], []
    # 모든 도메인 이미지 로드 및 누적
    for dom in domain_names:
        domain_dir = os.path.join(dir_path, dom)
        if not os.path.isdir(domain_dir):
            print(f"도메인 디렉토리 {domain_dir}가 없습니다. 건너뜁니다.")
            continue
        dataset = torchvision.datasets.ImageFolder(root=domain_dir, transform=transform)
        print(f"Loaded {dom}: {len(dataset)} images")
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=12)
        for imgs, lbls in tqdm(loader, desc=f"Processing {dom}"):
            all_images.append(imgs.cpu())
            all_labels.append(lbls.cpu())

    # 텐서 결합
    global_images = torch.cat(all_images, dim=0)
    global_labels = torch.cat(all_labels, dim=0)
    print(f"Combined dataset size: {global_images.size(0)} images")

    # alpha마다 파티셔닝 및 저장
    for alpha in [0.1, 0.3, 0.5]:
        print(f"--- alpha = {alpha} 처리 중 ---")
        X, y, statistic = separate_data(
            (global_images, global_labels), num_clients,
            num_classes, niid, balance, partition, 0, alpha
        )
        train_clients, test_clients = split_data(X, y)
        save_file(
            config_path, train_path, test_path,
            train_clients, test_clients,
            num_clients, num_classes, statistic,
            niid, balance, partition, start=0, alpha=alpha
        )
        print(f"alpha={alpha}용 데이터셋 저장 완료.\n")

    print("모든 도메인 혼합 후 데이터셋 생성 완료.")

if __name__ == "__main__":
    # 예: 클라이언트 수 10로 설정
    generate_office31_mixed(dir_path, num_clients=10, num_classes=num_classes, niid=True, balance=False, partition="dir")
