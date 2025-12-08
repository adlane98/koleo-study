# Introduction

Dans de nombreux problèmes de vision par ordinateur, l’objectif classique consiste à attribuer une étiquette de classe à chaque image : « chat », « chien », « voiture », etc. Dans cette étude, nous adoptons un point de vue différent : plutôt que de prédire directement une classe, nous cherchons à apprendre un espace d’embedding dans lequel les images d’objets similaires sont proches les unes des autres, et les images d’objets différents sont éloignées. Autrement dit, nous voulons apprendre une représentation continue qui capture la notion de similarité visuelle.

Pour construire un tel espace, nous nous appuyons sur l’architecture des réseaux siamois. Un réseau siamois ne se contente pas de traiter une image isolée : il compare plusieurs entrées en parallèle (typiquement un triplet ancre–positive–négative) et apprend à rapprocher les paires pertinentes tout en repoussant les paires non pertinentes. Pour la fonction de coût nous utiliserons une triplet loss. Celle-ci se chargera de formaliser précisément cet objectif en imposant une marge entre la distance ancre–positive et la distance ancre–négative dans l’espace des embeddings.

Nous introduisons ensuite la KoLeo loss, une régularisation destinée à mieux répartir la séparation entre classes. Nous visualiserons ceci grâce à une projection sur un espace 2D. La particularité de la KoLeo loss est qu’elle est dépendante du contenu du batch. En effet, les loss classiques sont généralement définies comme une moyenne sur le batch d’objets qui, pris individuellement, ne « voient » pas les autres exemples présents dans la même itération. La KoLeo loss, au contraire, exploite la distribution conjointe des embeddings d’un batch pour encourager un remplissage plus uniforme de l’espace et pénaliser les régions trop denses. Cette dépendance forte au batch rend particulièrement intéressante l’étude de l’accumulation de gradient, qui modifie la taille de batch effective et donc le comportement de cette régularisation. Mais est-ce réellement problématique d’avoir une loss aussi dépendante du batch, et dans quelle mesure cela impacte-t-il concrètement l’entraînement et les résultats obtenus ? C'est ce que nous allons essayer de comprendre.

L’article est structuré en trois grandes parties. Nous commençons par présenter le cadre des réseaux siamois et le modèle utilisé, ainsi que la triplet loss et la manière dont les triplets sont construits et évalués. Nous introduisons ensuite la KoLeo loss, expliquons son origine et son intégration dans notre pipeline, avant de comparer les résultats obtenus avec et sans cette régularisation. Enfin, nous discutons le rôle de l’accumulation de gradient dans ce contexte, en mettant en évidence la dépendance de la KoLeo loss à la taille de batch et l’impact que cela peut avoir sur la qualité finale des embeddings.


## Le dataset utilisé

Pour entraîner et évaluer notre modèle, nous utilisons le jeu de données [CIFAR‑10](https://www.cs.toronto.edu/~kriz/cifar.html), un benchmark classique composé de 60 000 images couleur de petite taille (32×32 pixels) réparties en 10 classes équilibrées : avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau et camion. Les images sont initialement séparées en 50 000 exemples d’entraînement et 10 000 exemples de test, mais dans notre cas nous utiliserons uniquement les données d'entrainement. 

Après téléchargement nous pouvons déjà remarquer que l'archive contient 6 fichiers binaires contenant les images: `data_batch_[1/2/3/4/5]` et `test_batch`. Nous utiliserons uniquement les fichiers commençant par data_batch.


```python
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1 = unpickle(f"../cifar-10-python/data_batch_1")
print(f"Clefs de data_batch : {batch_1.keys()}")
```

    Clefs de data_batch : dict_keys([b'batch_label', b'labels', b'data', b'filenames'])


Les clefs qui nous intéressent sont donc `data` et `labels`. Quelles sont leur types ?


```python
type(batch_1[b'data']), type(batch_1[b'labels'])
```




    (numpy.ndarray, list)



Quelles sont leurs tailles ?


```python
print(f"Taille de data : {batch_1[b'data'].shape}, Taille de labels : {len(batch_1[b'labels'])}")
```

    Taille de data : (10000, 3072), Taille de labels : 10000


Première particularité : la taille de la clef `data`. Chaque batch contient un tableau NumPy de forme `(10000, 3072)` : les 10 000 lignes correspondent aux images du batch, et les 3 072 colonnes à tous les pixels d’une image "aplatie" en un seul vecteur. En effet, une image CIFAR‑10 a une résolution de 32×32 pixels et 3 canaux couleur (R, G, B), soit \(32 x 32 x 3 = 3072\) valeurs au total.

Autrement dit, chaque ligne de `data` représente une image, mais sous forme linéarisée. Pour pouvoir afficher ou traiter ces images avec des bibliothèques de vision (et les passer à un réseau de neurones), nous devrons réorganiser les pixels dans une forme plus naturelle : `(32, 32, 3)`.

Mais avant cela, lisons tous les autres fichiers.


```python

data_batch = [unpickle(f"../cifar-10-python/data_batch_{i}") for i in range(1, 6)]
images = np.concatenate([data_batch[i][b'data'] for i in range(5)])
labels = np.concatenate([data_batch[i][b'labels'] for i in range(5)])

images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
images.shape
```




    (50000, 32, 32, 3)



Nous avons donc bien 50 000 images de taille 32x32 en RGB. En voici quelques unes sélectionnées aléatoirement avec leur labels.


```python
import matplotlib.pyplot as plt

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

n_rows, n_cols = 3, 5
indices = np.random.choice(len(images), n_rows * n_cols, replace=False)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

for ax, idx in zip(axes.ravel(), indices):
    ax.imshow(images[idx])
    label = label_names[labels[idx]]
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()
plt.show()

```


    
![png](article_files/article_9_0.png)
    


# Les réseaux siamois

## Qu'est-ce qu'un réseau siamois

Un réseau siamois est une architecture de réseau de neurones pensée non pas pour prédire directement une classe, mais pour comparer des exemples entre eux.

L’idée clé est la suivante : deux images que l’on considère comme « similaires » (par exemple deux chiens) doivent être proches l’une de l’autre dans cet espace, alors que deux images « différentes » (un chien et une voiture) doivent être éloignées. Pendant l’entraînement, on présente donc au modèle des paires ou des triplets d’images (ancre, positive, négative) et l’on ajuste les poids de façon à réduire la distance entre ancre et positive, tout en augmentant la distance entre ancre et négative.

Concrètement, dans le code, nous n’instancions qu’un seul réseau : nous lui passons tour à tour les images ancre, positive et négative, puis nous mettons à jour *une seule* fois les poids de ce modèle à partir de la loss calculée sur l’ensemble du triplet (nous aborderons la loss plus tard). Nous utiliserons ce modèle pour extraire les embeddings des images. Un *embedding* désigne ici la représentation vectorielle extraite par le réseau à partir d’une image, qui permet de comparer leur similarité dans un espace de dimension réduite. En pratique, presque n’importe quelle architecture de réseau de neurones (CNN, transformer, etc.) peut jouer ce rôle, à condition de produire un embedding en sortie.

Dans la suite, nous allons détailler le modèle utilisé pour produire ces embeddings, ainsi que la fonction de coût (triplet loss) qui formalise cette notion de similarité.

![Schéma d'un réseau siamois](siamese-scheme-french.png)

## VGG 11

VGG11 est une architecture de réseau de neurones convolutifs proposée en 2014 par une équipe de l’université d’Oxford (Simonyan et Zisserman). L’idée majeure de VGG est d’empiler de nombreux petits filtres convolutifs 3×3, séparés par des couches de pooling, plutôt que d’utiliser quelques grandes convolutions, ce qui permet d’augmenter la profondeur du réseau tout en gardant une structure très régulière.

VGG11 est l’une des variantes les plus simples de cette famille : onze couches organisées en blocs successifs, qui transforment une image RGB en un vecteur de caractéristiques de dimension fixe. Dans notre travail, nous partons de cette architecture pré‑entraînée sur ImageNet et nous l’adaptons pour produire des embeddings compacts adaptés à CIFAR‑10 et à l’apprentissage par triplets.

![VGG11](vgg11-model.png)

Mais voyons tout d'abord la taille du tenseur en sortie de VGG11.


```python
import torch
from torchvision import models

vgg = models.vgg11(pretrained=False)

sample = images[0]                              
x = torch.from_numpy(sample).permute(2, 0, 1)
x = x.unsqueeze(0).float() / 255.0

with torch.no_grad():
    out = vgg(x)

print("Taille du tenseur en sortie de VGG11 :", out.shape)
```
    Taille du tenseur en sortie de VGG11 : torch.Size([1, 1000])


La taille `(1, 1000)` signifie que, pour une image d’entrée, VGG11 renvoie un vecteur de 1 000 composantes. Ce nombre vient directement de la couche fully‑connected finale du modèle ImageNet d’origine : VGG11 a été conçu pour classer les images dans les 1 000 classes du challenge ImageNet, et sa dernière couche sort un vecteur de 1 000 probabilités, une probabilité pour chaque classe.

Dans notre cas, nous n’utiliserons pas ce vecteur. Nous utiliserons la sortie de la dernière couche de convolution de VGG donnée par `vgg.features`. Nous rajouterons juste une couche linéaire pour avoir un vecteur d'embedding de taille 128.

Quelle est la taille du tenseur juste après la dernière couche de convolution ?


```python
with torch.no_grad():
    out = vgg.features(x)

print("Taille du tenseur en sortie de la dernière couche de convolution de VGG11 :", out.shape)
```

    Taille du tenseur en sortie de la dernière couche de convolution de VGG11 : torch.Size([1, 512, 1, 1])


Nous remarquons que nous avons une taille de (512, 1, 1).
- 512 : le nombre de cartes de caractéristiques (features) produites par la dernière couche de convolution, donc 512 canaux différents décrivant l’image ;
- 1, 1 : la hauteur et la largeur spatiales, réduites à 1×1 par la succession de convolutions et de poolings, ce qui signifie que chaque canal résume toute l’image en un seul « neurone » (une valeur) avant de passer aux couches fully‑connected.

Nous aurons donc une taille en entrée de la couche linéaire de 512 x 1 x 1 = 512. Pour l'embedding, nous déciderons de réduire sa taille à 128 grâce à une couche linéaire (mais nous aurions très bien pu garder 512 et ne pas rajouter de couche linéaires !). A la fin du modèle nous rajoutons une couche de normalisation qui prendra tout son sens prochainement !


```python
import torch.nn as nn
import torch.nn.functional as F

class VGG11Embedding(nn.Module):
    def __init__(self, pretrained):
        super(VGG11Embedding, self).__init__()
        vgg = models.vgg11(pretrained=pretrained)
        self.features = vgg.features
        self.linear = nn.Linear(512, 128)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x
```

Vérifions la taille en sortie de notre modèle :


```python
vgg_embedding = VGG11Embedding(pretrained=False)

with torch.no_grad():
    out = vgg_embedding(x)

print("Taille du tenseur en sortie de notre modèle :", out.shape)
```

    Taille du tenseur en sortie de notre modèle : torch.Size([1, 128])


Nous avons donc bien un vecteur d'embedding de taille 128.

## La triplet loss

L’objectif de la triplet loss est d’imposer une structure géométrique à l’espace des embeddings : pour chaque triplet (ancre, positive, négative), on veut que l’image positive soit plus proche de l’ancre que l’image négative, avec une certaine marge. Autrement dit, on cherche à vérifier
\[ d(f(a), f(p)) + m < d(f(a), f(n)), \]
où \(f(\cdot)\) est le réseau d’embedding, \(d\) une mesure de distance (ou de "non‑similarité") et \(m > 0\) une marge fixée.


Dans notre cas, nous utilisons une distance dérivée de la similarité cosinus : plus deux vecteurs sont proches angulairement, plus ils sont considérés comme similaires. La triplet loss s’écrit alors, pour un batch de taille \(B\),
\[ \mathcal{L} = \frac{1}{B} \sum_{i=1}^B \max\big(0,\ d(a_i, p_i) - d(a_i, n_i) + m\big), \]
où chaque terme est nul dès que la contrainte est satisfaite (le triplet est "bon") et strictement positif lorsque l’ancre est encore trop proche de la négative.



```python
def triplet_loss(anchor, positive, negative, margin=0.4):
    positive_distances = 1 - F.cosine_similarity(anchor, positive, dim=1)
    negative_distances = 1 - F.cosine_similarity(anchor, negative, dim=1)
    loss = torch.clamp(positive_distances - negative_distances + margin, min=0)
    return loss.mean()
```





```python
triplets = np.empty((0, 3, 32, 32, 3), dtype=np.uint8)
triplets_labels = np.empty((0, 3), dtype=np.uint8)

for target in range(10):
    class_mask = (labels == target)
    images_target = images[class_mask]
    labels_target = labels[class_mask]

    pairs = images_target.reshape(-1, 2, 32, 32, 3)
    pos_labels = np.ones((len(pairs), 2), dtype=np.uint8) * target

    not_target_mask = (labels != target)
    images_not_target = images[not_target_mask]
    labels_not_target = labels[not_target_mask]

    # On echantillonne un nombre fixe de negatives pour cette classe
    n_neg = min(2500, len(images_not_target))
    neg_indices = np.random.choice(len(images_not_target), n_neg, replace=False)
    negatives = images_not_target[neg_indices]
    neg_labels = labels_not_target[neg_indices]

    pairs = pairs[:n_neg]
    pos_labels = pos_labels[:n_neg]

    class_triplets = np.concatenate(
        [pairs, negatives.reshape(n_neg, 1, 32, 32, 3)],
        axis=1,
    )
    class_triplet_labels = np.concatenate(
        [pos_labels, neg_labels.reshape(n_neg, 1)],
        axis=1,
    )

    triplets = np.concatenate([triplets, class_triplets], axis=0)
    triplets_labels = np.concatenate([triplets_labels, class_triplet_labels], axis=0)

triplets.shape, triplets_labels.shape

```




    ((25000, 3, 32, 32, 3), (25000, 3))



### La classe `TripletsCIFAR10Dataset` et la visualisation des triplets

Pour faciliter l’entraînement, nous encapsulons ces triplets dans un `Dataset` dédié. Le tenseur de triplets est converti au format `(N, 3, C, H, W)` puis, à chaque appel, on renvoie les trois images `(ancre, positive, négative)`. Nous pouvons ensuite prélever quelques indices au hasard dans ce Dataset pour visualiser des triplets concrets (images ancre/positive/négative et leurs labels associés) et vérifier que la construction est cohérente.


```python
from torch.utils.data import Dataset
import torchvision.transforms as T

class TripletsCIFAR10Dataset(Dataset):
    def __init__(self, triplets, transform=None):
        # (N, 3, H, W, C) -> (N, 3, C, H, W)
        self.triplets = torch.from_numpy(
            triplets.transpose(0, 1, 4, 2, 3) / 255.0
        ).float()
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

train_transforms = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

val_transforms = T.Compose([
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

triplet_dataset = TripletsCIFAR10Dataset(triplets)
len(triplet_dataset)
```




    25000



Voyons une sélection aléatoire de triplets.


```python
n_examples = 5
indices = np.random.choice(len(triplet_dataset), n_examples, replace=False)

fig, axes = plt.subplots(n_examples, 3, figsize=(6, 1 * n_examples))

for row_idx, idx in enumerate(indices):
    anchor, positive, negative = triplet_dataset[idx]

    anchor_img = anchor.permute(1, 2, 0).numpy()
    positive_img = positive.permute(1, 2, 0).numpy()
    negative_img = negative.permute(1, 2, 0).numpy()

    anchor_label = int(triplets_labels[idx, 0])
    positive_label = int(triplets_labels[idx, 1])
    negative_label = int(triplets_labels[idx, 2])

    axes[row_idx, 0].imshow(anchor_img)
    axes[row_idx, 0].set_title(f'anc: {label_names[anchor_label]}', fontsize=12)
    axes[row_idx, 0].axis('off')
    axes[row_idx, 0].set_ylabel(f'Triplet {idx}', fontsize=12, rotation=0, labelpad=50)

    axes[row_idx, 1].imshow(positive_img)
    axes[row_idx, 1].set_title(f'pos: {label_names[positive_label]}', fontsize=12)
    axes[row_idx, 1].axis('off')

    axes[row_idx, 2].imshow(negative_img)
    axes[row_idx, 2].set_title(f'neg: {label_names[negative_label]}', fontsize=12)
    axes[row_idx, 2].axis('off')

plt.tight_layout()
plt.show()

```


    
![png](article_files/article_28_0.png)
    



```python

```
