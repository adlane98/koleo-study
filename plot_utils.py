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
    show=True,
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
    if show:
        plt.show()
    plt.close()


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

def plot_distance_matrix_heatmap(dist_matrix, label_names, save_path=None, show=True):
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

    if show:
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


def plot_distance_matrix_and_embeddings(
    dist_matrix,
    embeddings_2d,
    ellipse_params,
    labels_array,
    label_names,
    save_path=None,
    title_suffix="",
    show=True,
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    col_labels = [f"{label_names[i]}" for i in range(len(label_names))]
    row_labels = [f"{label_names[i]}" for i in range(len(label_names))]
    
    sns.heatmap(
        dist_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': 'Cosine Distance'},
        ax=axes[0]
    )
    
    axes[0].set_title(f'Distance Matrix{title_suffix}', fontsize=14)
    axes[0].set_xticks(range(len(col_labels)))
    axes[0].set_xticklabels(col_labels, rotation=45, ha='right')
    axes[0].set_yticks(range(len(row_labels)))
    axes[0].set_yticklabels(row_labels, rotation=0)
    
    pca_2d_df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Label': labels_array,
        'Class': [label_names[int(label)] for label in labels_array]
    })
    
    palette = sns.color_palette("tab10", n_colors=10)
    class_names = sorted(pca_2d_df['Class'].unique())
    color_map = {cls: palette[i] for i, cls in enumerate(class_names)}
    
    sns.scatterplot(
        data=pca_2d_df, x='PC1', y='PC2',
        hue='Class', palette=color_map,
        alpha=0.7, s=30, ax=axes[1]
    )
    
    for cls in class_names:
        ep = ellipse_params[cls]
        center, w, h, angle = ep["center"], ep["width"], ep["height"], ep["angle"]
        color = color_map[cls]
        
        ellipse = Ellipse(
            xy=center, width=w, height=h, angle=angle,
            facecolor=(*color, 0.12), edgecolor=color, linewidth=2
        )
        axes[1].add_patch(ellipse)
    
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'PCA 2D Projection of Embeddings by Class{title_suffix}', fontsize=14)
    axes[1].legend(title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Combined plot saved at {save_path}")
    
    if show:
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


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_auc_and_ellipse_areas(
    results_negative,
    results_positive,
    categories,
    label_names,
):
    if label_names is None:
        label_names = list(results_negative["areas"].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: AUC Scores
    ax1 = axes[0]
    auc_negative = np.array(results_negative["auc"])
    auc_positive = np.array(results_positive["auc"])
    
    means = [auc_negative.mean(), auc_positive.mean()]
    stds = [auc_negative.std(), auc_positive.std()]
    
    bars = ax1.bar(categories, means, yerr=stds, capsize=8, 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('AUC Score', fontsize=12)
    ax1.set_title('AUC Scores Comparison', fontsize=14, fontweight='bold')
    y_min = min(means) - max(stds) - 0.01
    y_max = max(means) + max(stds) + 0.01
    ax1.set_ylim([y_min, y_max])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(i, mean + std + 0.001, f'{mean:.4f}\n±{std:.4f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Ellipse Areas
    ax2 = axes[1]
    
    classes_to_plot=["cat", "dog", "horse", "ship"]
    class_means_no = [np.array(results_negative["areas"][cls]).mean() for cls in classes_to_plot]
    class_stds_no = [np.array(results_negative["areas"][cls]).std() for cls in classes_to_plot]
    class_means_with = [np.array(results_positive["areas"][cls]).mean() for cls in classes_to_plot]
    class_stds_with = [np.array(results_positive["areas"][cls]).std() for cls in classes_to_plot]
    
    all_negative = np.array([np.mean(results_negative["areas"][cls]) for cls in label_names])
    all_positive = np.array([np.mean(results_positive["areas"][cls]) for cls in label_names])
    
    avg_mean_no = all_negative.mean()
    avg_std_no = all_negative.std()
    avg_mean_with = all_positive.mean()
    avg_std_with = all_positive.std()
    
    x_positions = np.arange(len(classes_to_plot) + 1)
    width = 0.35
    
    bars1 = ax2.bar(x_positions[:-1] - width/2, class_means_no, width, 
                    yerr=class_stds_no, capsize=5, label=categories[0],
                    color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x_positions[:-1] + width/2, class_means_with, width,
                    yerr=class_stds_with, capsize=5, label=categories[1],
                    color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1)
    
    bars3 = ax2.bar(x_positions[-1] - width/2, avg_mean_no, width,
                    yerr=avg_std_no, capsize=5, color='#3498db', alpha=0.7, 
                    edgecolor='black', linewidth=1, hatch='///')
    bars4 = ax2.bar(x_positions[-1] + width/2, avg_mean_with, width,
                    yerr=avg_std_with, capsize=5, color='#e74c3c', alpha=0.7,
                    edgecolor='black', linewidth=1, hatch='///')
    
    ax2.axvline(x=len(classes_to_plot) - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(classes_to_plot + ['Average'], fontsize=10)
    ax2.set_ylabel('Ellipse Area', fontsize=12)
    ax2.set_title('Ellipse Areas Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ax2.text(len(classes_to_plot) - 0.5, ax2.get_ylim()[1] * 0.95, '""', 
    #          ha='center', va='top', fontsize=10, fontweight='bold', 
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    plt.close()