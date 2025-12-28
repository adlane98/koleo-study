from datetime import datetime
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import models



def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_cifar10(data_path="../cifar-10-python"):
    data_batch = [unpickle(f"{data_path}/data_batch_{i}") for i in range(1, 6)]
    images = np.concatenate([data_batch[i][b'data'] for i in range(5)])
    labels = np.concatenate([data_batch[i][b'labels'] for i in range(5)])
    images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return images, labels


LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def build_triplets(images, labels, n_neg=2500, seed=None):
    triplets = np.empty((0, 3, 32, 32, 3), dtype=np.uint8)
    triplets_labels = np.empty((0, 3), dtype=np.uint8)

    if seed is not None:
        np.random.seed(seed)

    for target in range(10):
        class_mask = (labels == target)
        images_target = images[class_mask]

        pairs = images_target.reshape(-1, 2, 32, 32, 3)
        pos_labels = np.ones((len(pairs), 2), dtype=np.uint8) * target

        not_target_mask = (labels != target)
        images_not_target = images[not_target_mask]
        labels_not_target = labels[not_target_mask]

        n_neg_actual = min(n_neg, len(images_not_target))
        neg_indices = np.random.choice(len(images_not_target), n_neg_actual, replace=False)
        negatives = images_not_target[neg_indices]
        neg_labels = labels_not_target[neg_indices]

        pairs = pairs[:n_neg_actual]
        pos_labels = pos_labels[:n_neg_actual]

        class_triplets = np.concatenate([pairs, negatives.reshape(n_neg_actual, 1, 32, 32, 3)], axis=1)
        class_triplet_labels = np.concatenate([pos_labels, neg_labels.reshape(n_neg_actual, 1)], axis=1)

        triplets = np.concatenate([triplets, class_triplets], axis=0)
        triplets_labels = np.concatenate([triplets_labels, class_triplet_labels], axis=0)

    return triplets, triplets_labels


class TripletsCIFAR10Dataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = torch.from_numpy(triplets.transpose(0, 1, 4, 2, 3) / 255.0).float()
        self.transform = transform

    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor, positive, negative = triplet[0], triplet[1], triplet[2]
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative


TRAIN_TRANSFORMS = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

VAL_TRANSFORMS = T.Compose([
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])


class VGG11Embedding(nn.Module):
    def __init__(self, weights=None):
        super(VGG11Embedding, self).__init__()
        vgg = models.vgg11(weights=weights)
        self.features = vgg.features
        self.linear = nn.Linear(512, 128)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def triplet_loss(anchor, positive, negative, margin=0.4):
    positive_distances = 1 - F.cosine_similarity(anchor, positive, dim=1)
    negative_distances = 1 - F.cosine_similarity(anchor, negative, dim=1)
    loss = torch.clamp(positive_distances - negative_distances + margin, min=0)
    return loss.mean()


def create_datasets(triplets, triplets_labels, val_split=0.05, seed=None):
    num_train = int((1 - val_split) * len(triplets))
    
    if seed is not None:
        np.random.seed(seed)
    shuffle_indices = np.random.permutation(len(triplets))
    triplets = triplets[shuffle_indices]
    triplets_labels = triplets_labels[shuffle_indices]

    train_triplets = triplets[:num_train]
    val_triplets = triplets[num_train:]
    train_labels = triplets_labels[:num_train]
    val_labels = triplets_labels[num_train:]

    train_dataset = TripletsCIFAR10Dataset(train_triplets, transform=TRAIN_TRANSFORMS)
    val_dataset = TripletsCIFAR10Dataset(val_triplets, transform=VAL_TRANSFORMS)

    return train_dataset, val_dataset, val_triplets, val_labels


def setup_training_dir(runs_dir_name, config):
    runs_dir = Path(runs_dir_name)
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = runs_dir / timestamp
    save_dir.mkdir(exist_ok=True)

    with (save_dir / "config.json").open("w") as fp:
        json.dump(config, fp, indent=4)

    csv_headers = ["epoch", "train_loss", "val_loss", "simple_loss", "val_auc", "mean_positive_similarities",
                   "mean_negative_similarities", "mean_positive_euclidean_distances",
                   "mean_negative_euclidean_distances", "good_triplets_ratio"]
    
    metrics_path = save_dir / "training_metrics.csv"
    with open(metrics_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

    return save_dir, metrics_path, csv_headers


def log_metrics(metrics_path, csv_headers, epoch, train_loss, val_metrics):
    with open(metrics_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writerow({"epoch": epoch, "train_loss": train_loss, **val_metrics})

def print_metrics(val_metrics):
    metrics_str = ", ".join(
        f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in val_metrics.items()
    )
    print(f"Validation metrics — {metrics_str}")


def plot_losses(train_losses, val_losses, title="Evolution de la loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.show()


def construct_embeddings_by_class(net, labels, triplets, transforms, device):
    net.eval()

    embeddings_by_class = {i: [] for i in range(10)} 

    with torch.no_grad():
        anchor_labels = labels[:, 0]
        
        for idx in range(len(triplets)):
            label = int(anchor_labels[idx])
            
            img = torch.from_numpy(triplets[idx, 0].transpose(2, 0, 1) / 255.0).float()
            img = transforms(img)
            img = img.unsqueeze(0).to(device)
            
            embedding = net(img)
            embeddings_by_class[label].append(embedding.cpu())

    embeddings_by_class = {label: torch.cat(embeddings_by_class[label], dim=0) for label in range(10)}
    samples_per_class = [len(embeddings_by_class[i]) for i in range(10)]
    return embeddings_by_class