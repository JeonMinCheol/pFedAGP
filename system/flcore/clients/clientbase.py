import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from utils.data_utils import read_client_data

import umap
import matplotlib.pyplot as plt

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.num_workers = args.num_workers
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.dirchlet = args.dirchlet
        self.pt_path = f"../dataset/models/{self.dataset}/{self.algorithm}"
        self.plt_path = f"./embedding/{self.dataset}/{self.algorithm}/{self.dirchlet}/"

        if not os.path.exists(self.pt_path):
            os.mkdir(self.pt_path)

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # self.load_item(self.dirchlet, f"../../../dataset/models/{self.dataset}/{self.algorithm}/{str(self.dirchlet)}")


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=self.num_workers)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=self.num_workers)
    
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        return losses, train_num, train_correct

    def test_metrics(self, test_loader=None):
        # 1) 테스트 로더 준비
        testloader = test_loader if test_loader is not None else self.load_test_data()
        self.model.eval()

        # 2) 결과 저장용
        y_prob, y_true = [], []
        test_correct = 0
        test_num     = 0

        # 3) 로컬에서 본 라벨을 한 번만 수집해서 저장
        if not hasattr(self, 'seen_labels'):
            self.seen_labels = set()
            trainloader = self.load_train_data()
            for xb, yb in trainloader:
                # yb가 텐서인지 리스트인지 체크
                if isinstance(yb, torch.Tensor):
                    self.seen_labels.update(yb.cpu().tolist())
                else:
                    # 리스트 형태라면 각 텐서 원소를 꺼내서
                    for elem in yb:
                        self.seen_labels.update(elem.cpu().tolist())

        # 4) 테스트 루프
        with torch.no_grad():
            for x, y in testloader:
                # (a) x, y를 올바른 텐서로
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # (b) 학습하지 않은 라벨은 건너뛰기
                labels_cpu = y.cpu().tolist()
                mask = torch.tensor(
                    [lbl in self.seen_labels for lbl in labels_cpu],
                    dtype=torch.bool,
                    device=self.device
                )
                if mask.sum().item() == 0:
                    continue
                x, y = x[mask], y[mask]

                # (c) forward & 스코어
                output = self.model(x)                           # [b, C]
                probs  = F.softmax(output, dim=1).cpu().numpy()  # [b, C]
                truths = label_binarize(
                    y.cpu().numpy(),
                    classes=np.arange(self.num_classes)
                )                                              # [b, C]

                # (d) accuracy 집계
                test_correct += (output.argmax(dim=1) == y).sum().item()
                test_num     += y.size(0)

                y_prob.append(probs)
                y_true.append(truths)

        # 5) 최종 concatenate
        if y_prob:
            y_prob = np.concatenate(y_prob, axis=0)  # (N_test, C)
            y_true = np.concatenate(y_true, axis=0)
        else:
            # 전혀 평가할 샘플 없으면 빈 배열
            y_prob = np.zeros((0, self.num_classes))
            y_true = np.zeros((0, self.num_classes))

        return test_correct, test_num, y_prob, y_true


    def visualize_umap(self, embeddings, labels):
        """
        Args:
            embeddings (Tensor): [N, D] - 전체 임베딩
            labels (Tensor): [N] - 정수형 클래스 라벨
        """
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings.cpu().numpy())

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
        plt.title(f"Client {self.id}: Embeddings")
        plt.grid(True)
        plt.tight_layout()
        return plt

    def visualize_embeddings(self):
        """
        모델 임베딩을 추출하고 UMAP 시각화하는 함수
        """
        trainloader = self.load_train_data()
        self.model.eval()

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x = [x[0].to(self.device)]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)  # [B, D]
                all_embeddings.append(rep)
                all_labels.append(y)

        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
        all_labels = torch.cat(all_labels, dim=0)          # [N]

        plt = self.visualize_umap(all_embeddings, all_labels)
        os.makedirs(self.plt_path, exist_ok=True)
        plt.savefig(self.plt_path + f"Client {self.id} Embedding UMAP")

    # def visualize_multi_protos_with_labels(self):
    #     """
    #     K개의 프로토타입 전체 + 임베딩을 UMAP으로 시각화하고 클래스 ID도 표시
    #     """
    #     trainloader = self.load_train_data()
    #     self.model.eval()

    #     all_embeddings = []
    #     all_labels = []

    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if isinstance(x, list):
    #                 x = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)

    #             rep = self.model.base(x)
    #             all_embeddings.append(rep)
    #             all_labels.append(y)

    #     all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    #     all_labels = torch.cat(all_labels, dim=0)          # [N]

    #     # 모든 프로토타입 수집
    #     proto_points = []
    #     proto_labels = []
    #     for cls, protos in self.multi_protos.items():  # protos: [K, D]
    #         if protos is None or protos.numel() == 0:
    #             continue

    #         for k in range(protos.size(0)):
    #             proto_points.append(protos[k].detach().cpu().numpy())
    #             proto_labels.append(cls)

    #     if len(proto_points) == 0:
    #         print(f"[Client {self.id}] Warning: proto_points is empty. Skipping visualization.")
    #         return

    #     # UMAP 차원 축소: fit → 임베딩 + 프로토타입 동시에
    #     proto_points = np.stack(proto_points)
    #     proto_labels = np.array(proto_labels)

    #     reducer = umap.UMAP(n_components=2, random_state=42)
    #     reduced_protos = reducer.fit_transform(proto_points)

    #     # 시각화
    #     plt.figure(figsize=(10, 8))

    #     # K Prototypes per class
    #     scatter = plt.scatter(reduced_protos[:, 0], reduced_protos[:, 1],
    #                 c=proto_labels, cmap='tab10', s=150, edgecolor='black', label="Prototypes")

    #     # 텍스트로 class ID 표시
    #     for i, (x, y) in enumerate(reduced_protos):
    #         plt.text(x + 0.5, y, str(proto_labels[i]), fontsize=8, color='black', weight='bold')

    #     plt.colorbar(scatter, label='Class Label')
    #     plt.title(f"Client {self.id}: Prototypes")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(self.plt_path + f"Client {self.id} Prototypes")