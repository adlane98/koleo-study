import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop, SGD
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

from training_utils import (
    KoLeoLoss, build_triplets, create_datasets, 
    get_device, load_cifar10, print_metrics, 
    triplet_loss, validation_loop
    )

images, labels = load_cifar10("../cifar-10-python")

seed = 42
np.random.seed(seed)
triplets, triplets_labels = build_triplets(images, labels, n_neg=2500, seed=seed)
train_dataset, val_dataset, val_triplets, val_labels = create_datasets(triplets, triplets_labels, val_split=0.05, seed=seed)

koleo_loss_fn = KoLeoLoss()

EPOCHS = 15

class VGG11Embedding(nn.Module):
    def __init__(self, embedding_size, weights=None):
        super(VGG11Embedding, self).__init__()
        vgg = models.vgg11(weights=weights)
        self.features = vgg.features
        self.linear = nn.Linear(512, embedding_size)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def train_loop(model, grad_accum_steps, dataloader, optimizer, margin, koleo_weight, print_freq=100):
    model.train()
    loss_accum = 0.0
    epoch_loss = 0.0
    device = get_device()

    for batch_idx, (anc, pos, neg) in tqdm(enumerate(dataloader)):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        anc_feat, pos_feat, neg_feat = model(anc), model(pos), model(neg)

        t_loss = triplet_loss(anc_feat, pos_feat, neg_feat, margin)
        if koleo_weight is not None:
            all_embeddings = torch.cat([anc_feat, pos_feat, neg_feat], dim=0)
            k_loss = koleo_loss_fn(all_embeddings)
            loss = t_loss + koleo_weight * k_loss
        else:
            loss = t_loss
        
        loss = loss / grad_accum_steps
        loss_accum += loss.item()
        epoch_loss += loss.item()
        loss.backward()

        if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % print_freq == 0:
            print(f"Batch {batch_idx+1}: Loss = {loss_accum / print_freq:.4f}")
            loss_accum = 0.0

    return epoch_loss / (batch_idx + 1)

def run_experiment(
    trial, 
    margin,
    koleo_weight,
    grad_accum_steps,
    batch_size,
    learning_rate,
    embedding_size,
    optimizer_name
):
    device = get_device()
    model = VGG11Embedding(embedding_size, weights=models.VGG11_Weights.IMAGENET1K_V1).to(device)

    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"Optimizer need to be Adam, RMSprop or SGD")

    effective_batch_size = batch_size // grad_accum_steps
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)

    auc = 0
    best_auc = 0.0
    for epoch_idx in range(EPOCHS):
        train_loss = train_loop(model, grad_accum_steps, train_loader, optimizer, margin, koleo_weight)
        val_metrics = validation_loop(model, val_loader, margin, koleo_weight, device)
            
        print(f"Epoch {epoch_idx+1} - train_loss: {train_loss:.4f}, val_loss: {val_metrics['val_loss']:.4f}, val_auc: {val_metrics['val_auc']:.4f}")
        print_metrics(val_metrics)

        auc = val_metrics['val_auc']

        if auc > best_auc:
            best_auc = val_metrics['val_auc']
            print(f"New best AUC: {best_auc:.4f} at epoch {epoch_idx+1}")

        trial.report(auc, epoch_idx + 1)

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch_idx+1} with AUC: {auc:.4f}")
            raise optuna.exceptions.TrialPruned()
        
    print(f"Trial {trial.number} completed with AUC: {auc:.4f}")
    return auc


def objective(trial):
    triplet_loss_margin = trial.suggest_float("triplet_loss_margin", 0.2, 0.6)

    use_koleo = trial.suggest_categorical("use_koleo", [True, False])
    if use_koleo:
        koleo_weight = trial.suggest_float("koleo_weight", 1e-4, 1e-1, log=True)
        trial.set_user_attr("koleo_weight", koleo_weight)
    else:
        koleo_weight = None
        trial.set_user_attr("koleo_weight", 0.0)
    
    grad_accum_steps = trial.suggest_categorical("grad_accum_steps", [1, 2, 4])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    embedding_size = trial.suggest_categorical("embedding_size", [64, 128, 256])
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    print("--------------------------------")
    print(f"Trial {trial.number}: triplet_loss_margin: {triplet_loss_margin}, use_koleo: {use_koleo}, koleo_weight: {koleo_weight}, grad_accum_steps: {grad_accum_steps}, batch_size: {batch_size}, learning_rate: {learning_rate}, embedding_size: {embedding_size}, optimizer: {optimizer}")

    auc_score = run_experiment(
        trial,
        triplet_loss_margin,
        koleo_weight,
        grad_accum_steps,
        batch_size,
        learning_rate,
        embedding_size,
        optimizer
    )

    return auc_score


def main():
    storage = optuna.storages.RDBStorage("sqlite:///optuna_study.db")
    study = optuna.create_study(study_name="koleo_gradacc", direction="maximize", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.TrialState.COMPLETE]

    print("Study statistics:")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of completed trials: ", len(completed_trials))
    print("  Best trial: ")
    print("    Value: ", study.best_trial.value)
    print("    Params: ")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")

if __name__ == "__main__":
    main()
