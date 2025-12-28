from pathlib import Path

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

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


def get_ellipse_params_per_class(embeddings_2d, labels_array, label_names, coverage):
    ellipse_params = {}
    for cls in label_names:
        cls_idx = label_names.index(cls)
        X = embeddings_2d[labels_array == cls_idx]
        center, width, height, angle = compute_ellipse_parameters(X, coverage)
        ellipse_params[cls] = {
            "center": center.tolist(),
            "width": float(width),
            "height": float(height),
            "angle": float(angle),
        }
    return ellipse_params


def plot_embeddings_with_ellipses(
    embeddings_2d,
    ellipse_params,
    labels_array,
    label_names,
    save_img_path,
):
    pca_2d_df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Label': labels_array,
        'Class': [label_names[int(label)] for label in labels_array]
    })

    fig, ax = plt.subplots(figsize=(12, 10))
    
    palette = sns.color_palette("tab10", n_colors=10)
    class_names = sorted(pca_2d_df['Class'].unique())
    color_map = {cls: palette[i] for i, cls in enumerate(class_names)}

    sns.scatterplot(
        data=pca_2d_df, x='PC1', y='PC2',
        hue='Class', palette=color_map,
        alpha=0.7, s=30, ax=ax
    )

    for cls in class_names:
        ep = ellipse_params[cls]
        center, w, h, angle = ep["center"], ep["width"], ep["height"], ep["angle"]
        color = color_map[cls]
        
        ellipse = Ellipse(
            xy=center, width=w, height=h, angle=angle,
            facecolor=(*color, 0.12), edgecolor=color, linewidth=2
        )
        ax.add_patch(ellipse)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA 2D Projection of Embeddings by Class')
    ax.legend(title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_distance_matrix(embeddings_by_class):
    dist_matrix = np.zeros((10, 10))

    for i in range(10):
        for j in range(10):
            emb_i = embeddings_by_class[i]
            emb_j = embeddings_by_class[j]
            
            emb_i_norm = F.normalize(emb_i, p=2, dim=1)
            emb_j_norm = F.normalize(emb_j, p=2, dim=1)
            
            cosine_sim = torch.mm(emb_i_norm, emb_j_norm.t())
            cosine_dist = 1 - cosine_sim
            
            dist_matrix[i, j] = cosine_dist.mean().item()
    
    return dist_matrix

def plot_distance_matrix_heatmap(dist_matrix, label_names, save_path=None):
    plt.figure(figsize=(10, 9))

    col_labels = [f"{label_names[i]}" for i in range(len(label_names))]
    row_labels = [f"{label_names[i]}" for i in range(len(label_names))]

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

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Distance matrix heatmap saved at {save_path}")

    plt.show()
    plt.close()

    same_class_dists = np.diag(dist_matrix)
    diff_class_dists = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j:
                diff_class_dists.append(dist_matrix[i, j])

    print(f"\nIntra-class distance: mean={np.mean(same_class_dists):.4f}, std={np.std(same_class_dists):.4f}")
    print(f"Inter-class distance: mean={np.mean(diff_class_dists):.4f}, std={np.std(diff_class_dists):.4f}")
    print(f"Separation margin: {np.mean(diff_class_dists) - np.mean(same_class_dists):.4f}")

