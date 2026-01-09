import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from torchvision.models import VGG11_Weights

from koleo_loss import KoLeoLoss
from model import VGG11Embedding
from plot_utils import (
    compute_distance_matrix,
    get_ellipse_params_per_class,
    plot_distance_matrix_and_embeddings,
    plot_distance_matrix_heatmap,
    plot_embeddings_with_ellipses
)
from training_utils import (
    LABEL_NAMES,
    TRAIN_TRANSFORMS,
    VAL_TRANSFORMS,
    build_triplets,
    construct_embeddings_by_class,
    get_device,
    load_cifar10,
    plot_losses,
    print_metrics,
    triplet_loss,
    TripletsCIFAR10Dataset,
    validation_loop
)


def train_loop(net, dataloader, optimizer, margin, device, koleo_weight=None, koleo_loss_fn=None, print_freq=100):
    net.train()
    loss_accum = 0.0
    epoch_loss = 0.0
    
    for batch_idx, (anc, pos, neg) in enumerate(dataloader):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        anc_feat, pos_feat, neg_feat = net(anc), net(pos), net(neg)

        loss = triplet_loss(anc_feat, pos_feat, neg_feat, margin)
        
        if koleo_weight is not None and koleo_loss_fn is not None:
            all_embeddings = torch.cat([anc_feat, pos_feat, neg_feat], dim=0)
            k_loss = koleo_loss_fn(all_embeddings)
            loss = loss + koleo_weight * k_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()
        epoch_loss += loss.item()

        if (batch_idx + 1) % print_freq == 0:
            print(f"Batch {batch_idx+1}: Loss = {loss_accum / print_freq:.4f}")
            loss_accum = 0.0

    return epoch_loss / (batch_idx + 1)


def setup_experiment(config):
    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = runs_dir / timestamp
    save_dir.mkdir(exist_ok=True)

    with (save_dir / "config.json").open("w") as fp:
        json.dump(config, fp, indent=4)

    csv_headers = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_auc",
        "mean_positive_similarities",
        "mean_negative_similarities",
        "mean_positive_euclidean_distances",
        "mean_negative_euclidean_distances",
        "good_triplets_ratio"
    ]
    
    metrics_path = save_dir / "training_metrics.csv"
    with open(metrics_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

    return save_dir, metrics_path, csv_headers


def log_metrics(metrics_path, csv_headers, epoch, train_loss, val_metrics):
    with open(metrics_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writerow({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics['val_loss'],
            "val_auc": val_metrics['val_auc'],
            "mean_positive_similarities": val_metrics['mean_positive_similarities'],
            "mean_negative_similarities": val_metrics['mean_negative_similarities'],
            "mean_positive_euclidean_distances": val_metrics['mean_positive_euclidean_distances'],
            "mean_negative_euclidean_distances": val_metrics['mean_negative_euclidean_distances'],
            "good_triplets_ratio": val_metrics['good_triplets_ratio']
        })


def generate_visualizations(net, val_triplets, val_labels, save_dir, device):
    print("\nGenerating visualizations...")
    
    embeddings_by_class = construct_embeddings_by_class(
        net, val_labels, val_triplets, VAL_TRANSFORMS, device
    )
    
    dist_matrix = compute_distance_matrix(embeddings_by_class)
    plot_distance_matrix_heatmap(
        dist_matrix, 
        LABEL_NAMES, 
        save_dir / "distance_matrix_heatmap.png",
        show=False
    )
    
    all_embeddings = torch.cat([embeddings_by_class[k] for k in embeddings_by_class], dim=0)
    
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(all_embeddings)
    embeddings_2d = (embeddings_2d - embeddings_2d.min(axis=0)) / (
        embeddings_2d.max(axis=0) - embeddings_2d.min(axis=0)
    )
    
    samples_per_class = [len(embeddings_by_class[i]) for i in range(10)]
    labels_array = np.concatenate([
        np.full(count, label) for label, count in enumerate(samples_per_class)
    ])
    
    ellipse_params = get_ellipse_params_per_class(
        embeddings_2d, labels_array, LABEL_NAMES, coverage=0.5
    )
    
    plot_embeddings_with_ellipses(
        embeddings_2d,
        ellipse_params,
        labels_array,
        LABEL_NAMES,
        save_img_path=save_dir / "embeddings_2d_pca.png",
        show=False
    )
    
    plot_distance_matrix_and_embeddings(
        dist_matrix,
        embeddings_2d,
        ellipse_params,
        labels_array,
        LABEL_NAMES,
        save_path=save_dir / "combined_visualization.png",
        show=False
    )
    
    same_class_dists = np.diag(dist_matrix)
    diff_class_dists = []
    for i in range(10):
        for j in range(10):
            if i != j:
                diff_class_dists.append(dist_matrix[i, j])
    
    stats = {
        "intra_class_distance_mean": float(np.mean(same_class_dists)),
        "intra_class_distance_std": float(np.std(same_class_dists)),
        "inter_class_distance_mean": float(np.mean(diff_class_dists)),
        "inter_class_distance_std": float(np.std(diff_class_dists)),
        "separation_margin": float(np.mean(diff_class_dists) - np.mean(same_class_dists))
    }
    
    with open(save_dir / "embedding_statistics.json", "w") as fp:
        json.dump(stats, fp, indent=4)
    
    print(f"\nIntra-class distance: mean={stats['intra_class_distance_mean']:.4f}, "
          f"std={stats['intra_class_distance_std']:.4f}")
    print(f"Inter-class distance: mean={stats['inter_class_distance_mean']:.4f}, "
          f"std={stats['inter_class_distance_std']:.4f}")
    print(f"Separation margin: {stats['separation_margin']:.4f}")


def load_config_from_json(config_path, epochs=None):
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if epochs is not None:
        print(f"Overriding epochs from {config.get('epochs', 'N/A')} to {epochs}")
        config['epochs'] = epochs
    
    print("\nLoaded configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config


def load_best_config_from_optuna(db_path="optuna_study.db", study_name="koleo_gradacc", epochs=15):
    print("Loading best hyperparameters from Optuna study...")
    
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    best_trial = study.best_trial
    
    print(f"\nBest trial: #{best_trial.number}")
    print(f"  Best AUC: {best_trial.value:.4f}")
    print(f"  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    koleo_weight = None
    if best_trial.params.get("use_koleo", False):
        koleo_weight = best_trial.params.get("koleo_weight", 0.1)
    
    seed = 42
    best_config = {
        "seed": seed,
        "batch_size": best_trial.params["batch_size"],
        "learning_rate": best_trial.params["learning_rate"],
        "epochs": epochs,
        "margin": best_trial.params["triplet_loss_margin"],
        "embedding_size": best_trial.params["embedding_size"],
        "optimizer": best_trial.params["optimizer"],
        "val_split": 0.05,
        "use_koleo": best_trial.params["use_koleo"],
        "koleo_weight": koleo_weight,
        "grad_accum_steps": best_trial.params["grad_accum_steps"],
        "model_name": "TripletVGG11",
        "dataset": "CIFAR-10",
        "optuna_trial_number": best_trial.number,
        "optuna_best_auc": best_trial.value,
        "note": f"Best hyperparameters from Optuna study '{study_name}' ({len(study.trials)} trials)"
    }
    
    return best_config


def main(epochs=2, config_source="optuna_study.db"):
    print("="*70)
    print("Training TripletVGG11")
    print("="*70)
    
    config_source_path = Path(config_source)
    
    if config_source_path.suffix == '.json':
        print("\nLoading from config file...")
        best_config = load_config_from_json(config_source, epochs=epochs)
    elif config_source_path.suffix == '.db' or not config_source_path.suffix:
        print("\nLoading from Optuna study...")
        best_config = load_best_config_from_optuna(
            db_path=config_source,
            study_name="koleo_gradacc",
            epochs=epochs
        )
        print("\n" + "="*70)
        print("Configuration:")
        print("="*70)
        print(json.dumps(best_config, indent=2))
        print("="*70)
    else:
        raise ValueError(f"Unsupported config source format: {config_source}. "
                        f"Please provide either a .db (Optuna) or .json (config) file.")
    

    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    print("\nLoading CIFAR-10 dataset...")
    images, labels = load_cifar10("../cifar-10-python")
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    print("\nBuilding triplets...")
    np.random.seed(seed)
    triplets, triplets_labels = build_triplets(images, labels, n_neg=2500, seed=seed)
    print(f"Triplets shape: {triplets.shape}")
    
    val_split = best_config['val_split']
    num_train = int((1 - val_split) * len(triplets))
    
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(len(triplets))
    shuffled_triplets = triplets[shuffle_indices]
    shuffled_triplets_labels = triplets_labels[shuffle_indices]
    
    train_triplets = shuffled_triplets[:num_train]
    val_triplets = shuffled_triplets[num_train:]
    val_labels = shuffled_triplets_labels[num_train:]
    
    train_dataset = TripletsCIFAR10Dataset(train_triplets, transform=TRAIN_TRANSFORMS)
    val_dataset = TripletsCIFAR10Dataset(val_triplets, transform=VAL_TRANSFORMS)
    
    print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")
    
    save_dir, metrics_path, csv_headers = setup_experiment(best_config)
    print(f"\nExperiment directory: {save_dir}")
    
    gt = torch.Generator()
    gt.manual_seed(seed)
    
    batch_size = best_config['batch_size']
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gt)
    
    print("\nInitializing model...")
    embedding_size = best_config['embedding_size']
    print(f"Embedding size: {embedding_size}")
    net = VGG11Embedding(embedding_size=embedding_size, weights=VGG11_Weights.IMAGENET1K_V1).to(device)
    
    optimizer_name = best_config['optimizer']
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=best_config['learning_rate'])
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=best_config['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    koleo_weight = best_config.get('koleo_weight')
    koleo_loss_fn = None
    if koleo_weight is not None:
        koleo_loss_fn = KoLeoLoss()
        print(f"Using KoLeo loss with weight: {koleo_weight}")
    else:
        print("Training without KoLeo loss")
    
    print("\nEvaluating before training...")
    val_metrics = validation_loop(net, val_loader, best_config['margin'], koleo_weight, device)
    print(f"Before training:")
    print_metrics(val_metrics)
    log_metrics(metrics_path, csv_headers, 0, "", val_metrics)
    
    train_losses = []
    val_losses = []
    best_auc = 0
    best_epoch = 0
    best_epoch_path = None
    
    epochs = best_config['epochs']
    print(f"\nStarting training for {epochs} epochs...")
    print("="*70)
    
    for epoch_idx in range(epochs):
        print(f"\nEpoch {epoch_idx + 1}/{epochs}")
        print("-" * 70)
        
        train_loss = train_loop(
            net, train_loader, optimizer, best_config['margin'], device, 
            koleo_weight=koleo_weight, koleo_loss_fn=koleo_loss_fn, print_freq=100
        )
        val_metrics = validation_loop(net, val_loader, best_config['margin'], koleo_weight, device)
        
        val_losses.append(val_metrics['val_loss'])
        train_losses.append(train_loss)
        
        print(f"\nEpoch {epoch_idx + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val AUC: {val_metrics['val_auc']:.4f}")
        print_metrics(val_metrics)
        
        log_metrics(metrics_path, csv_headers, epoch_idx + 1, train_loss, val_metrics)
        
        if val_metrics['val_auc'] > best_auc:
            best_auc = val_metrics['val_auc']
            best_epoch = epoch_idx + 1
            
            if best_epoch_path is not None:
                best_epoch_path.unlink()
            
            best_epoch_path = save_dir / f'best_model_epoch_{epoch_idx + 1}_auc_{best_auc:.4f}.pth'
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': best_auc,
                'val_metrics': val_metrics,
                'config': best_config
            }, best_epoch_path)
            
            print(f"\n  ✓ New best AUC: {best_auc:.4f} (saved to {best_epoch_path.name})")
        
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'auc': val_metrics['val_auc'],
            'config': best_config
        }, save_dir / f'checkpoint_epoch_{epoch_idx + 1}.pth')
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"Best model saved at: {best_epoch_path}")
    
    print("\nPlotting training curves...")
    plot_losses(train_losses, val_losses, title="TripletVGG11 Training - Best Config", show=False)
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_epoch_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    generate_visualizations(net, val_triplets, val_labels, save_dir, device)
    
    final_summary = {
        "model_name": "TripletVGG11",
        "training_completed": datetime.now().isoformat(),
        "best_epoch": best_epoch,
        "best_auc": float(best_auc),
        "total_epochs": epochs,
        "config": best_config,
        "experiment_dir": str(save_dir)
    }
    
    with open(save_dir / "final_summary.json", "w") as fp:
        json.dump(final_summary, fp, indent=4)
    
    print("\n" + "="*70)
    print("All results saved to:", save_dir)
    print("="*70)
    print("\nGenerated files:")
    for file in sorted(save_dir.glob("*")):
        print(f"  - {file.name}")
    print("\n✓ Training complete!")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Train TripletVGG11"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="optuna_study.db",
        help="Path to config source: either Optuna database (.db) or config file (.json) (default: optuna_study.db)"
    )
    
    args = parser.parse_args()
    
    main(epochs=args.epochs, config_source=args.config)
