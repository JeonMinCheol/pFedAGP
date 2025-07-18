import torch
import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from PIL import UnidentifiedImageError

# https://github.com/FengHZ/KD3A/blob/master/datasets/DomainNet.py
def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)

            if not path.exists(data_path):
                continue

            try:
                # 이미지 유효성 체크
                with Image.open(data_path) as img:
                    img.verify()  # format만 검사 (load보다 빠름)
                data_paths.append(data_path)
                data_labels.append(label)
            except (UnidentifiedImageError, OSError) as e:
                print(f"[Skip] Corrupted image: {data_path}")
                continue
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(dataset_path, domain_name):
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "DomainNet/"
dir_path = "DomainNet/"

# Allocate data to users
def generate_DomainNet(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata"
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    urls = [
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip', 
    ]
    http_head = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'

    # Get DomainNet data
    if not os.path.exists(root):
        os.makedirs(root)

    # for d, u in zip(domains, urls):
    #     os.system(f'wget -q {u} -P {root}')
    #     os.system(f'unzip {root}/{d}.zip -d {root} -q')
    #     os.system(f'wget {http_head}domainnet/txt/{d}_train.txt -P {root}/splits')
    #     os.system(f'wget {http_head}domainnet/txt/{d}_test.txt -P {root}/splits')
    #     pass

    X, y = [], []
    for d in domains:
        print(f"domain name: {d}")
        train_loader, test_loader = get_domainnet_dloader(root, d)

        dataset_image, dataset_label = [], []

        for imgs, labels in train_loader:
            for img, label in zip(imgs, labels):
                dataset_image.append(img.numpy())
                dataset_label.append(label.item())

        for imgs, labels in test_loader:
            for img, label in zip(imgs, labels):
                dataset_image.append(img.numpy())
                dataset_label.append(label.item())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))

    for alpha in [0.1, 0.3, 0.5]:
        train_data, test_data = split_data(X, y)
        # modify the code in YOUR_ENV/lib/python3.8/site-packages/numpy/lib Line #678 from protocol=3 to protocol=4
        save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
            statistic, True, False, "dir", 0, alpha)

generate_DomainNet(dir_path)