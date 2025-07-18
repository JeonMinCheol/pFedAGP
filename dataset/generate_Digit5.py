import numpy as np
import os
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file, separate_data
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict

data_path = "Digit5/raw"
dir_path = "Digit5/"

# https://github.com/FengHZ/KD3A/blob/master/datasets/DigitFive.py
def load_mnist(base_path):
    print("load mnist")
    mnist_train_data = loadmat(path.join(base_path, "mnist32_60_10_train.mat"))
    mnist_test_data  = loadmat(path.join(base_path, "mnist32_60_10_test.mat"))

    # 원본 X shape = (32,32,1,N) → (N,32,32,1)
    mnist_train = np.reshape(mnist_train_data['X'], (-1, 32, 32, 1))
    mnist_test  = np.reshape(mnist_test_data['X'], (-1, 32, 32, 1))

    # 3채널, (N, C, H, W) 변환 후 10% 자르기
    mnist_train = np.concatenate([mnist_train]*3, axis=3)\
                     .transpose(0,3,1,2).astype(np.float32)
    mnist_test  = np.concatenate([mnist_test]*3,  axis=3)\
                     .transpose(0,3,1,2).astype(np.float32)

    num_train = int(len(mnist_train) * 0.01)
    num_test  = 9000

    mnist_train = mnist_train[:num_train]
    mnist_test  = mnist_test[:num_test]

    # 레이블도 동일한 num_train, num_test 기준으로 자르기
    labels_train_mat = mnist_train_data['y']    # shape (N,1) 혹은 (1,N)
    labels_test_mat  = mnist_test_data['y']

    # reshape/argmax 등 전처리
    train_label = np.argmax(labels_train_mat[:num_train], axis=1)
    test_label  = np.argmax(labels_test_mat[:num_test],  axis=1)

    # shuffle
    inds = np.random.permutation(num_train)
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]

    return mnist_train, train_label, mnist_test, test_label


def load_mnist_m(base_path):
    print("load mnist_m")
    data = loadmat(path.join(base_path, "mnistm32_60_10_train.mat"))
    test = loadmat(path.join(base_path, "mnistm32_60_10_test.mat"))

    # 원본
    X_train_raw = data['X']       # shape (N, H, W, C)
    y_train_raw = data['y'].reshape(-1)  # shape (N,)
    X_test_raw  = test['X']
    y_test_raw  = test['y'].reshape(-1)

    # 슬라이싱 비율
    num_train = int(len(X_train_raw) * 0.01)
    num_test  = 9000

    # 이미지 전처리
    X_train = X_train_raw.transpose(0,3,1,2).astype(np.float32)[:num_train]
    X_test  = X_test_raw.transpose(0,3,1,2).astype(np.float32)[:num_test]

    # 레이블도 동일 개수만큼 자르기
    y_train = y_train_raw[:num_train]
    y_test  = y_test_raw[:num_test]

    # shuffle
    inds = np.random.permutation(num_train)
    X_train = X_train[inds]
    y_train = y_train[inds]

    return X_train, y_train, X_test, y_test

def load_svhn(base_path):
    print("load svhn")
    train_mat = loadmat(path.join(base_path, "svhn_train.mat"))
    test_mat  = loadmat(path.join(base_path, "svhn_test.mat"))

    X_train_raw = train_mat['X']
    y_train_raw = train_mat['y'].reshape(-1)
    X_test_raw  = test_mat['X']
    y_test_raw  = test_mat['y'].reshape(-1)

    num_train = int(X_train_raw.shape[0] * 0.01)
    num_test  = 9000

    # Transpose 및 슬라이싱
    X_train = X_train_raw.transpose(0,3,1,2).astype(np.float32)[:num_train]
    X_test  = X_test_raw.transpose(0,3,1,2).astype(np.float32)[:num_test]

    y_train = y_train_raw[:num_train]
    y_test  = y_test_raw[:num_test]

    # 10 → 0 매핑
    y_train[y_train == 10] = 0
    y_test [y_test  == 10] = 0

    inds = np.random.permutation(num_train)
    X_train = X_train[inds]
    y_train = y_train[inds]

    return X_train, y_train, X_test, y_test


def load_syn(base_path):
    print("load syn")
    train_mat = loadmat(path.join(base_path, "syn32_train.mat"))
    test_mat  = loadmat(path.join(base_path, "syn32_test.mat"))

    X_train_raw = train_mat['X']
    y_train_raw = train_mat['y'].reshape(-1)
    X_test_raw  = test_mat['X']
    y_test_raw  = test_mat['y'].reshape(-1)

    num_train = int(X_train_raw.shape[0] * 0.01)
    num_test  = 9000

    X_train = X_train_raw.transpose(0,3,1,2).astype(np.float32)[:num_train]
    X_test  = X_test_raw.transpose(0,3,1,2).astype(np.float32)[:num_test]

    y_train = y_train_raw[:num_train]
    y_test  = y_test_raw[:num_test]

    # 10 → 0 매핑
    y_train[y_train == 10] = 0
    y_test [y_test  == 10] = 0

    inds = np.random.permutation(num_train)
    X_train = X_train[inds]
    y_train = y_train[inds]

    return X_train, y_train, X_test, y_test


def load_usps(base_path):
    print("load usps")
    train_mat = loadmat(path.join(base_path, "usps32_train.mat"))
    test_mat  = loadmat(path.join(base_path, "usps32_test.mat"))

    X_train_raw = train_mat['X']
    y_train_raw = train_mat['y'].reshape(-1)
    X_test_raw  = test_mat['X']
    y_test_raw  = test_mat['y'].reshape(-1)

    ratio = 0.3
    num_train = int(X_train_raw.shape[0] * ratio)
    num_test  = 9000

    X_train = X_train_raw[:num_train].transpose(0,3,1,2).astype(np.float32)
    X_test  = X_test_raw[:num_test].transpose(0,3,1,2).astype(np.float32)

    y_train = y_train_raw[:num_train]
    y_test  = y_test_raw[:num_test]

    y_train[y_train == 10] = 0
    y_test [y_test  == 10] = 0

    inds = np.random.permutation(num_train)
    X_train = X_train[inds]
    y_train = y_train[inds]

    return X_train, y_train, X_test, y_test


class Digit5Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(Digit5Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            if img_np.shape[0] == 1 and (img_np.shape[1] == 1 or img_np.shape[2] == 1):
                img_np = img_np.reshape(32, 32)

            elif img_np.shape[0] == 1:
                img_np = img_np[0]  # (1, H, W) → (H, W)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.labels.shape[0]

def digit5_dataset_read(base_path, domain):
    if domain == "mnist":
        train_image, train_label, test_image, test_label = load_mnist(base_path)
    elif domain == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m(base_path)
    elif domain == "svhn":
        train_image, train_label, test_image, test_label = load_svhn(base_path)
    elif domain == "syn":
        train_image, train_label, test_image, test_label = load_syn(base_path)
    elif domain == "usps":
        train_image, train_label, test_image, test_label = load_usps(base_path)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # raise train and test data loader
    train_dataset = Digit5Dataset(data=train_image, labels=train_label, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = Digit5Dataset(data=test_image, labels=test_label, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader

def generate_Digit5_mixed(dir_path, data_path, total_clients, num_classes,
                           niid=True, balance=False, partition="dir", class_per_client=2):
    # 디렉토리 생성
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, "train/")
    test_path  = os.path.join(dir_path, "test/")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path,  exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")

    # 도메인 리스트
    domain_names = ['mnist', 'syn', 'usps', 'svhn', 'mnistm']

    # 모든 도메인 데이터 누적
    all_images = []
    all_labels = []
    for d in domain_names:
        train_loader, test_loader = digit5_dataset_read(data_path, d)
        imgs_train, lbls_train = next(iter(train_loader))
        imgs_test,  lbls_test  = next(iter(test_loader))
        imgs   = np.concatenate([imgs_train.numpy(),  imgs_test.numpy()], 0)
        labels = np.concatenate([lbls_train.numpy(), lbls_test.numpy()], 0)
        print(f"Loaded {d}: {imgs.shape[0]} samples")
        all_images.append(imgs)
        all_labels.append(labels)

    # 전체 데이터 결합
    global_images = np.concatenate(all_images, axis=0)
    global_labels = np.concatenate(all_labels, axis=0)
    print(f"Combined dataset: {global_images.shape[0]} samples")

    # 알파별 파티셔닝 및 저장
    for alpha in [0.5]:
        print(f"--- alpha = {alpha} 진행 중 ---")
        X, y, statistic = separate_data(
            (global_images, global_labels), total_clients, num_classes,
            niid, balance, partition,
            class_per_client=class_per_client, alpha=alpha
        )
        train_clients, test_clients = split_data(X, y)
        save_file(
            config_path, train_path, test_path,
            train_clients, test_clients,
            total_clients, num_classes, statistic,
            niid, balance, partition,
            start=0, alpha=alpha
        )
        print(f"alpha={alpha}용 데이터셋 저장 완료.\n")

    print("모든 도메인 혼합 후 Digit5 데이터셋 생성 완료.")

if __name__ == "__main__":
    # 예시 호출
    generate_Digit5_mixed(
        dir_path='./Digit5/',
        data_path='./Digit5/raw',
        total_clients=10,
        num_classes=10
    )
 