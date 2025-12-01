# Introduction
But de l'étude, le chemin suivi
Présenter l'objectif : on veut apprendre au modèle à identifier des objets similaires. Différent d'attribuer une classe à chaque objet.
Présenter la loss que l'on va utiliser triplet loss
Notre but est de construire des embeddings/projections d'une image sur un espace de plus petite dimension.

Etude sous l'angle des siamese networks

En trois parties
- Siamese network
- Introduction de la Koleo Loss
- Effet de la gradient accumulation sur la Koleo Loss

## Le dataset utilisé
CIPHAR10
présenter le dataset et le format avec des petits schémas

# Siamese network

## Qu'est-ce qu'un réseau siamois

## Le modèle que l'on va utiliser 
VGG11 et sa redéfinition

## Définition de la triplet loss
Expliquer la notion de margin

## La construction des triplets 
### définition de la classe dataset
### Les transformations utilisées

## Le training
### paramètres du training
### Métriques utilisées pour la performance
### train loop
### validation loop
### training

## Résultats en utilisant les embeddings
### Matrice de distance
### Moyenne des distances entre les ancres et les images positives/négatives
### Courbe ROC et valeur AUC
### PCA
#### PCA 2D
Montrer à quoi ressemble la PCA sans normalisation toute seule
Normalisation
Calcul des ellipses
Calcul de l'aire des ellipses

# La koleo loss

## Explication de la KoLeo loss et d'où elle vient
## Intégration de la koleo loss dans le code
## Réentrainement avec la koleo loss
## Résultats de l'entrainement
## Résultats de l'aire des ellipses avec la kfold

# TODO - L'accumulation de gradient
## A quoi sert l'accumulation de gradient
## Pourquoi la KoLeo loss est dépendante du batch
## Réentrainement avec l'accumulation de gradient
## Résultats

# Références
Site du siamese network
article de la koleo loss dinov2 ou v3 je sais plus
ciphar10