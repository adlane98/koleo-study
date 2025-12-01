import csv
from datetime import datetime
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T
import yaml

from koleo_loss import KoLeoLoss


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_cifar10_data(cifar_path):
    data_batch = [unpickle(f"{cifar_path}/data_batch_{i}") for i in range(1, 6)]
    images = np.concatenate([data_batch[i][b'data'] for i in range(5)])
    labels = np.concatenate([data_batch[i][b'labels'] for i in range(5)])
    images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return images, labels


def create_triplets(images, labels, seed=42):
    np.random.seed(seed)
    triplets = np.empty((0, 3, 32, 32, 3), dtype=np.uint8)
    triplets_labels = np.empty((0, 3), dtype=np.uint8)

    for target in range(10):
        triplet_target = images[np.where(target == labels)].reshape(-1, 2, 32, 32, 3)
        triplet_not_target = images[np.where(target != labels)]
        pos_labels = np.ones([len(triplet_target), 2]) * target
        neg_labels = labels[np.where(target != labels)]

        random_indices = np.random.choice(len(triplet_not_target), 2500)
        triplet_not_target = triplet_not_target[random_indices]
        neg_labels = neg_labels[random_indices]
        target_labels = np.concatenate([pos_labels, neg_labels[:, None]], axis=1)

        triplet_target = np.concatenate(
            [triplet_target, triplet_not_target.reshape(2500, 1, 32, 32, 3)],
            axis=1
        )

        triplets = np.concatenate([triplets, triplet_target], axis=0)
        triplets_labels = np.concatenate([triplets_labels, target_labels], axis=0)
    
    return triplets, triplets_labels


class TripletsCIFAR10Dataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = torch.from_numpy(triplets.transpose((0, 1, 4, 2, 3)) / 255).float()
        self.transform = transform

    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor, positive, negative = triplet[0], triplet[1], triplet[2]
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative


class VGG11Embedding(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=128):
        super(VGG11Embedding, self).__init__()
        weights = models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg11(weights=weights)
        self.features = vgg.features
        self.linear = nn.Linear(512, embedding_dim)
        
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


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def compute_ellipse_parameters(embeddings, coverage):
    center = np.median(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    d2 = np.einsum("ij,jk,ik->i", embeddings - center, inv_cov, embeddings - center)
    threshold = np.quantile(d2, coverage)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2.0 * np.sqrt(vals * threshold)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    return center, width, height, angle


class Training:
    def __init__(self, config, train_triplets, val_triplets, train_labels, val_labels, 
                 run_dir, fold_idx, device):
        self.config = config
        self.train_triplets = train_triplets
        self.val_triplets = val_triplets
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.run_dir = run_dir
        self.fold_idx = fold_idx
        self.device = device
        
        self.label_names = config.get('label_names', 
            ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
        self.mean = config['normalization']['mean']
        self.std = config['normalization']['std']
        self.coverage = config.get('ellipse_coverage', 0.50)
        
        self.koleo_loss = KoLeoLoss()
        self.metrics = {}

    def _create_dataloaders(self):
        train_transforms = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        val_transforms = T.Compose([T.Normalize(mean=self.mean, std=self.std)])
        
        train_dataset = TripletsCIFAR10Dataset(self.train_triplets, transform=train_transforms)
        val_dataset = TripletsCIFAR10Dataset(self.val_triplets, transform=val_transforms)
        
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def _create_model(self):
        self.net = VGG11Embedding(
            pretrained=self.config['model']['pretrained'],
            embedding_dim=self.config['model']['embedding_dim']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), 
            lr=self.config['training']['learning_rate']
        )

    def _train_epoch(self, epoch_idx):
        self.net.train()
        loss_accum = 0
        epoch_loss = 0
        margin = self.config['training']['margin']
        koleo_coeff = self.config['training'].get('koleo_loss_coeff')
        print_freq = self.config['training'].get('print_freq', 100)
        
        for batch_idx, (anc, pos, neg) in enumerate(self.train_loader):
            anc, pos, neg = anc.to(self.device), pos.to(self.device), neg.to(self.device)
            anc_feat, pos_feat, neg_feat = self.net(anc), self.net(pos), self.net(neg)

            loss = triplet_loss(anc_feat, pos_feat, neg_feat, margin)
            if koleo_coeff is not None:
                loss += koleo_coeff * self.koleo_loss(torch.cat([anc_feat, pos_feat, neg_feat], dim=0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_accum += loss.item()
            epoch_loss += loss.item()

            if (batch_idx + 1) % print_freq == 0:
                print(f"  Fold {self.fold_idx} - Batch {batch_idx+1}: Loss = {loss_accum / print_freq:.4f}")
                loss_accum = 0

        return epoch_loss / (batch_idx + 1)

    def _validate(self):
        self.net.eval()
        val_loss = 0
        good_triplets = 0
        total_triplets = 0
        margin = self.config['training']['margin']
        koleo_coeff = self.config['training'].get('koleo_loss_coeff')

        positive_similarities = []
        negative_similarities = []
        positive_euclidean_distances = []
        negative_euclidean_distances = []

        with torch.no_grad():
            for batch_idx, (anc, pos, neg) in enumerate(self.val_loader):
                anc, pos, neg = anc.to(self.device), pos.to(self.device), neg.to(self.device)
                anc_feat, pos_feat, neg_feat = self.net(anc), self.net(pos), self.net(neg)

                loss = triplet_loss(anc_feat, pos_feat, neg_feat, margin)
                if koleo_coeff is not None:
                    loss += koleo_coeff * self.koleo_loss(torch.cat([anc_feat, pos_feat, neg_feat], dim=0))

                val_loss += loss.item()

                batch_pos_euc = F.pairwise_distance(anc_feat, pos_feat, p=2)
                batch_neg_euc = F.pairwise_distance(anc_feat, neg_feat, p=2)
                positive_euclidean_distances.append(batch_pos_euc)
                negative_euclidean_distances.append(batch_neg_euc)

                batch_pos_sim = F.cosine_similarity(anc_feat, pos_feat, dim=1)
                batch_neg_sim = F.cosine_similarity(anc_feat, neg_feat, dim=1)
                positive_similarities.append(batch_pos_sim)
                negative_similarities.append(batch_neg_sim)

                good_triplets += (batch_pos_sim > batch_neg_sim).sum()
                total_triplets += anc.shape[0]

        positive_euclidean_distances = torch.cat(positive_euclidean_distances, dim=0)
        negative_euclidean_distances = torch.cat(negative_euclidean_distances, dim=0)
        positive_similarities = torch.cat(positive_similarities, dim=0)
        negative_similarities = torch.cat(negative_similarities, dim=0)

        predict_similarities = torch.cat([positive_similarities, negative_similarities], dim=0)
        target_similarities = torch.cat([
            torch.ones_like(positive_similarities), 
            torch.zeros_like(negative_similarities)
        ], dim=0)

        val_auc = roc_auc_score(
            target_similarities.cpu().numpy(), 
            predict_similarities.cpu().numpy()
        )

        return {
            'val_loss': val_loss / (batch_idx + 1),
            'val_auc': val_auc,
            'mean_positive_similarities': positive_similarities.mean().item(),
            'mean_negative_similarities': negative_similarities.mean().item(),
            'mean_positive_euclidean_distances': positive_euclidean_distances.mean().item(),
            'mean_negative_euclidean_distances': negative_euclidean_distances.mean().item(),
            'good_triplets_ratio': (good_triplets / total_triplets).item(),
            'positive_similarities': positive_similarities.cpu(),
            'negative_similarities': negative_similarities.cpu()
        }

    def _compute_embeddings_by_class(self):
        self.net.eval()
        val_transform = T.Compose([T.Normalize(mean=self.mean, std=self.std)])
        
        embeddings_by_class = {i: [] for i in range(10)}
        anchor_labels = self.val_labels[:, 0]

        with torch.no_grad():
            for idx in range(len(self.val_triplets)):
                label = int(anchor_labels[idx])
                img = torch.from_numpy(
                    self.val_triplets[idx, 0].transpose(2, 0, 1) / 255.0
                ).float()
                img = val_transform(img).unsqueeze(0).to(self.device)
                embedding = self.net(img)
                embeddings_by_class[label].append(embedding.cpu())

        embeddings_by_class = {
            label: torch.cat(embeddings_by_class[label], dim=0) 
            for label in range(10) if embeddings_by_class[label]
        }
        samples_per_class = {i: len(embeddings_by_class.get(i, [])) for i in range(10)}
        
        return embeddings_by_class, samples_per_class

    def _compute_distance_matrix(self, embeddings_by_class):
        dist_matrix = np.zeros((10, 10))
        
        for i in range(10):
            for j in range(10):
                if i not in embeddings_by_class or j not in embeddings_by_class:
                    dist_matrix[i, j] = np.nan
                    continue
                    
                emb_i = embeddings_by_class[i]
                emb_j = embeddings_by_class[j]
                
                emb_i_norm = F.normalize(emb_i, p=2, dim=1)
                emb_j_norm = F.normalize(emb_j, p=2, dim=1)
                
                cosine_sim = torch.mm(emb_i_norm, emb_j_norm.t())
                cosine_dist = 1 - cosine_sim
                
                dist_matrix[i, j] = cosine_dist.mean().item()
        
        return dist_matrix

    def _save_distance_matrix(self, dist_matrix):
        plt.figure(figsize=(10, 9))
        col_labels = [self.label_names[i] for i in range(10)]
        row_labels = [self.label_names[i] for i in range(10)]

        sns.heatmap(
            dist_matrix, 
            xticklabels=col_labels, 
            yticklabels=row_labels,
            annot=True,
            fmt='.2f',
            cmap='viridis', 
            cbar_kws={'label': 'Cosine Distance'}
        )

        plt.title('Distance Matrix', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.run_dir / "distance_matrix_heatmap.png")
        plt.close()

        np.save(self.run_dir / "distance_matrix.npy", dist_matrix)

    def _save_roc_curve(self, positive_similarities, negative_similarities):
        predict = torch.cat([positive_similarities, negative_similarities]).numpy()
        target = np.concatenate([
            np.ones(len(positive_similarities)), 
            np.zeros(len(negative_similarities))
        ])

        fpr, tpr, _ = roc_curve(target, predict)
        auc = roc_auc_score(target, predict)

        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], c='violet', ls='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right", fontsize=15)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver Operating Characteristic curve', weight='bold', fontsize=15)
        plt.savefig(self.run_dir / "roc_curve.png")
        plt.close()

        return auc

    def _compute_and_save_pca(self, embeddings_by_class, samples_per_class):
        present_classes = [c for c in range(10) if c in embeddings_by_class and len(embeddings_by_class[c]) > 0]
        all_embeddings = torch.cat([embeddings_by_class[k] for k in present_classes], dim=0).numpy()
        labels_array = np.concatenate([
            np.full(samples_per_class[c], c) for c in present_classes
        ])

        pca_3d = PCA(n_components=3)
        embeddings_3d = pca_3d.fit_transform(all_embeddings)
        np.save(self.run_dir / "embeddings_3d_pca.npy", embeddings_3d)

        pca_2d = PCA(n_components=2)
        embeddings_2d = pca_2d.fit_transform(all_embeddings)
        
        embeddings_2d_normalized = (embeddings_2d - embeddings_2d.min(axis=0)) / (
            embeddings_2d.max(axis=0) - embeddings_2d.min(axis=0) + 1e-8
        )
        np.save(self.run_dir / "embeddings_2d_pca.npy", embeddings_2d_normalized)

        return embeddings_2d_normalized, labels_array, present_classes

    def _compute_and_save_ellipses(self, embeddings_2d, labels_array, present_classes):
        ellipse_params = {}
        
        for cls_idx in present_classes:
            cls_name = self.label_names[cls_idx]
            X = embeddings_2d[labels_array == cls_idx]
            
            if len(X) < 3:
                continue
                
            center, width, height, angle = compute_ellipse_parameters(X, self.coverage)
            area = np.pi * width * height
            ellipse_params[cls_name] = {
                "center": center.tolist(),
                "width": float(width),
                "height": float(height),
                "angle": float(angle),
                "area": float(area)
            }

        areas = [ep["area"] for ep in ellipse_params.values()]
        mean_area = np.mean(areas) if areas else 0
        median_area = np.median(areas) if areas else 0

        ellipse_results = {
            "ellipse_params": ellipse_params,
            "mean_area": float(mean_area),
            "median_area": float(median_area)
        }
        
        with open(self.run_dir / "ellipse.json", "w") as fp:
            json.dump(ellipse_results, fp, indent=4)

        return ellipse_params, mean_area, median_area

    def _save_2d_plot(self, embeddings_2d, labels_array, ellipse_params, present_classes):
        pca_2d_df = pd.DataFrame({
            'PC1': embeddings_2d[:, 0],
            'PC2': embeddings_2d[:, 1],
            'Label': labels_array,
            'Class': [self.label_names[int(label)] for label in labels_array]
        })

        fig = px.scatter(
            pca_2d_df, x='PC1', y='PC2',
            color='Class',
            color_discrete_sequence=px.colors.qualitative.T10,
            hover_data={'Label': True, 'Class': True, 'PC1': ':.3f', 'PC2': ':.3f'},
            title='PCA 2D Projection of Embeddings by Class (Normalized)',
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(
            xaxis_title='PC1',
            yaxis_title='PC2',
            width=1000,
            height=800,
            legend=dict(title='Classes')
        )

        class_to_color = {
            tr.legendgroup: tr.marker.color
            for tr in fig.data
            if getattr(tr, "mode", "") == "markers" and tr.legendgroup
        }

        for cls_idx in present_classes:
            cls_name = self.label_names[cls_idx]
            if cls_name not in ellipse_params:
                continue
                
            ep = ellipse_params[cls_name]
            center, w, h, angle = ep["center"], ep["width"], ep["height"], ep["angle"]

            t = np.linspace(0, 2 * np.pi, 200)
            ex, ey = (w / 2) * np.cos(t), (h / 2) * np.sin(t)
            c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            x = c * ex - s * ey + center[0]
            y = s * ex + c * ey + center[1]

            col = class_to_color.get(cls_name, px.colors.qualitative.T10[0])
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=None,
                legendgroup=str(cls_name),
                showlegend=False,
                line=dict(color=col, width=2),
                fill='toself',
                fillcolor=hex_to_rgba(col, 0.12),
                hoverinfo='skip'
            ))

        fig.write_image(str(self.run_dir / "embeddings_2d_with_ellipses.png"), scale=2, width=1000, height=800)

    def _save_embeddings_and_labels(self, embeddings_by_class, samples_per_class):
        embeddings_data = {
            str(k): v.numpy().tolist() for k, v in embeddings_by_class.items()
        }
        with open(self.run_dir / "embeddings_by_class.json", "w") as fp:
            json.dump(embeddings_data, fp)

        with open(self.run_dir / "samples_per_class.json", "w") as fp:
            json.dump(samples_per_class, fp)

        np.save(self.run_dir / "val_labels.npy", self.val_labels)

    def run(self):
        print(f"\n{'='*60}")
        print(f"Starting training for Fold {self.fold_idx}")
        print(f"Run directory: {self.run_dir}")
        print(f"{'='*60}")

        self._create_dataloaders()
        self._create_model()

        epochs = self.config['training']['epochs']
        save_weights = self.config['saving'].get('save_weights', False)
        save_freq = self.config['saving'].get('save_every_n_epochs', 1)

        metrics_path = self.run_dir / "training_metrics.csv"
        csv_headers = [
            "epoch", "train_loss", "val_loss", "val_auc",
            "mean_positive_similarities", "mean_negative_similarities",
            "mean_positive_euclidean_distances", "mean_negative_euclidean_distances",
            "good_triplets_ratio"
        ]

        with open(metrics_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()

        best_auc = 0
        best_val_metrics = None

        for epoch_idx in range(epochs):
            train_loss = self._train_epoch(epoch_idx)
            val_metrics = self._validate()

            print(f"  Fold {self.fold_idx} - Epoch {epoch_idx+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"AUC: {val_metrics['val_auc']:.4f}")

            with open(metrics_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                writer.writerow({
                    "epoch": epoch_idx + 1,
                    "train_loss": train_loss,
                    "val_loss": val_metrics['val_loss'],
                    "val_auc": val_metrics['val_auc'],
                    "mean_positive_similarities": val_metrics['mean_positive_similarities'],
                    "mean_negative_similarities": val_metrics['mean_negative_similarities'],
                    "mean_positive_euclidean_distances": val_metrics['mean_positive_euclidean_distances'],
                    "mean_negative_euclidean_distances": val_metrics['mean_negative_euclidean_distances'],
                    "good_triplets_ratio": val_metrics['good_triplets_ratio']
                })

            if val_metrics['val_auc'] > best_auc:
                best_auc = val_metrics['val_auc']
                best_val_metrics = val_metrics
                if save_weights:
                    torch.save(self.net.state_dict(), self.run_dir / f'best_epoch_{epoch_idx+1}.pth')

            if save_weights and (epoch_idx + 1) % save_freq == 0:
                torch.save(self.net.state_dict(), self.run_dir / f'epoch_{epoch_idx+1}.pth')

        embeddings_by_class, samples_per_class = self._compute_embeddings_by_class()
        dist_matrix = self._compute_distance_matrix(embeddings_by_class)
        self._save_distance_matrix(dist_matrix)

        auc = self._save_roc_curve(
            best_val_metrics['positive_similarities'],
            best_val_metrics['negative_similarities']
        )

        embeddings_2d, labels_array, present_classes = self._compute_and_save_pca(
            embeddings_by_class, samples_per_class
        )
        ellipse_params, mean_area, median_area = self._compute_and_save_ellipses(
            embeddings_2d, labels_array, present_classes
        )
        self._save_2d_plot(embeddings_2d, labels_array, ellipse_params, present_classes)
        self._save_embeddings_and_labels(embeddings_by_class, samples_per_class)

        same_class_dists = np.diag(dist_matrix)
        valid_same_class = same_class_dists[~np.isnan(same_class_dists)]
        
        diff_class_dists = []
        for i in range(10):
            for j in range(10):
                if i != j and not np.isnan(dist_matrix[i, j]):
                    diff_class_dists.append(dist_matrix[i, j])
        diff_class_dists = np.array(diff_class_dists)

        self.metrics = {
            'fold_idx': self.fold_idx,
            'auc': auc,
            'mean_ellipse_area': mean_area,
            'median_ellipse_area': median_area,
            'mean_anchor_positive_distance': best_val_metrics['mean_positive_euclidean_distances'],
            'mean_anchor_negative_distance': best_val_metrics['mean_negative_euclidean_distances'],
            'mean_same_class_distance': float(np.mean(valid_same_class)) if len(valid_same_class) > 0 else np.nan,
            'std_same_class_distance': float(np.std(valid_same_class)) if len(valid_same_class) > 0 else np.nan,
            'mean_diff_class_distance': float(np.mean(diff_class_dists)) if len(diff_class_dists) > 0 else np.nan,
            'std_diff_class_distance': float(np.std(diff_class_dists)) if len(diff_class_dists) > 0 else np.nan,
            'separation_margin': float(np.mean(diff_class_dists) - np.mean(valid_same_class)) if len(diff_class_dists) > 0 and len(valid_same_class) > 0 else np.nan,
            'good_triplets_ratio': best_val_metrics['good_triplets_ratio'],
            'final_val_loss': best_val_metrics['val_loss']
        }

        with open(self.run_dir / "final_metrics.json", "w") as fp:
            json.dump(self.metrics, fp, indent=4)

        return self.metrics


class KFoldCrossValidation:
    def __init__(self, config_path):
        with open(config_path, 'r') as fp:
            self.config = yaml.safe_load(fp)
        
        self.device = get_device()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = Path('runs') / self.timestamp
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.base_dir / "config.json", "w") as fp:
            json.dump(self.config, fp, indent=4)

        self.all_metrics = []

    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def run(self):
        print(f"Starting K-Fold Cross-Validation")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.base_dir}")
        print(f"Config: {json.dumps(self.config, indent=2)}")

        seed = self.config['data'].get('seed', 42)
        self._set_seed(seed)

        cifar_path = self.config['data']['cifar_path']
        images, labels = load_cifar10_data(cifar_path)
        triplets, triplets_labels = create_triplets(images, labels, seed=seed)

        shuffle_indices = np.random.permutation(len(triplets))
        triplets = triplets[shuffle_indices]
        triplets_labels = triplets_labels[shuffle_indices]

        n_splits = self.config['kfold']['n_splits']
        shuffle = self.config['kfold'].get('shuffle', True)
        
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(triplets)):
            run_dir = self.base_dir / f"run_{fold_idx}"
            run_dir.mkdir(exist_ok=True)

            train_triplets = triplets[train_idx]
            val_triplets = triplets[val_idx]
            train_labels = triplets_labels[train_idx]
            val_labels = triplets_labels[val_idx]

            trainer = Training(
                config=self.config,
                train_triplets=train_triplets,
                val_triplets=val_triplets,
                train_labels=train_labels,
                val_labels=val_labels,
                run_dir=run_dir,
                fold_idx=fold_idx,
                device=self.device
            )

            metrics = trainer.run()
            self.all_metrics.append(metrics)

        self._compile_results()
        self._print_summary()

    def _compile_results(self):
        csv_path = self.base_dir / "kfold_metrics_summary.csv"
        
        if self.all_metrics:
            fieldnames = list(self.all_metrics[0].keys())
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for metrics in self.all_metrics:
                    writer.writerow(metrics)

    def _print_summary(self):
        if not self.all_metrics:
            print("No metrics collected.")
            return

        summary_lines = []
        summary_lines.append("\n" + "="*80)
        summary_lines.append("K-FOLD CROSS-VALIDATION SUMMARY")
        summary_lines.append("="*80)

        metric_keys = [
            'auc', 'mean_ellipse_area', 'median_ellipse_area',
            'mean_anchor_positive_distance', 'mean_anchor_negative_distance',
            'mean_same_class_distance', 'std_same_class_distance',
            'mean_diff_class_distance', 'std_diff_class_distance',
            'separation_margin', 'good_triplets_ratio', 'final_val_loss'
        ]

        summary_data = {}
        for key in metric_keys:
            values = [m[key] for m in self.all_metrics if not np.isnan(m.get(key, np.nan))]
            if values:
                summary_data[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        summary_lines.append(f"\nNumber of folds: {len(self.all_metrics)}")
        summary_lines.append(f"\nMetrics Summary (mean ± std [min, max]):")
        summary_lines.append("-" * 60)

        for key, stats in summary_data.items():
            summary_lines.append(
                f"  {key:35s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )

        summary_lines.append("\n" + "-"*60)
        summary_lines.append("KEY PERFORMANCE INDICATORS:")
        summary_lines.append("-"*60)
        
        if 'auc' in summary_data:
            summary_lines.append(f"  Average AUC: {summary_data['auc']['mean']:.4f} ± {summary_data['auc']['std']:.4f}")
        if 'separation_margin' in summary_data:
            summary_lines.append(f"  Average Separation Margin: {summary_data['separation_margin']['mean']:.4f} ± {summary_data['separation_margin']['std']:.4f}")
        if 'good_triplets_ratio' in summary_data:
            summary_lines.append(f"  Average Good Triplets Ratio: {summary_data['good_triplets_ratio']['mean']:.4f} ± {summary_data['good_triplets_ratio']['std']:.4f}")

        summary_lines.append("="*80)

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        with open(self.base_dir / "summary.txt", "w") as fp:
            fp.write(summary_text)

        with open(self.base_dir / "summary_stats.json", "w") as fp:
            json.dump(summary_data, fp, indent=4)

        print(f"\nResults saved to: {self.base_dir}")
        print(f"  - kfold_metrics_summary.csv: All metrics for each fold")
        print(f"  - summary.txt: Human-readable summary")
        print(f"  - summary_stats.json: Statistical summary as JSON")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    kfold_cv = KFoldCrossValidation(args.config)
    kfold_cv.run()


if __name__ == '__main__':
    main()

