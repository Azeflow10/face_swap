# TP2 — Détection de visages, masques et Face Swap

Ce projet a pour but de comprendre et mettre en pratique les principes de base de la détection de visages et d’yeux à l’aide d’OpenCV, ainsi que la manipulation d’images via des masques. À la fin, vous serez capable de :
- Détecter automatiquement des visages et des yeux dans une image.
- Créer des masques elliptiques simples et “intelligents” centrés sur les yeux.
- Réaliser un échange de visages réaliste (face swap) avec adoucissement des bords.

---

## Arborescence du projet

```
face_swap_tp2/
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── verifier_environnement.py
├── images/
│   ├── img_test.jpg
│   ├── img1.jpg
│   ├── img2.jpg
│   └── deux_visages.jpg        ← image avec deux visages visibles
└── scripts/
    ├── detect_yeux.py
    ├── etape_masque.py         ← TP2.1 : masque simple sur visage
    ├── etape_masque_repris.py  ← TP2.2 : masque intelligent centré sur les yeux
    └── face_swap.py            ← TP2.3 : échange de visages (face swap)
```

---

## Prérequis

1) Python + dépendances
- Python 3.x
- OpenCV et NumPy

Installation rapide:
```bash
pip install opencv-python numpy
```

2) Fichiers de classificateurs Haar
- haarcascade_frontalface_default.xml
- haarcascade_eye.xml

Vous pouvez les récupérer depuis le dépôt officiel OpenCV:
- Classificateurs Haar Cascade: https://github.com/opencv/opencv/tree/master/data/haarcascades

Placez ces fichiers à la racine du projet comme dans l’arborescence ci-dessus.

---

## Vérification de l’environnement

Avant de commencer, exécutez:
```bash
python verifier_environnement.py
```

Ce script vérifie:
- La présence des fichiers HaarCascade (haarcascade_frontalface_default.xml, haarcascade_eye.xml)
- La présence du dossier images/
- Que les images sont bien chargeables

---

## Étapes du TP

### 🧩 Étape 1 — Création d’un masque elliptique simple
Script: `scripts/etape_masque.py`

Objectif:
- Détecter un visage dans une image.
- Créer un masque elliptique centré sur le visage.
- Appliquer ce masque pour ne garder que la zone du visage.

Exécution:
```bash
python scripts/etape_masque.py
```

Résultat attendu (affichage en fenêtres):
- Le visage détecté
- Le masque elliptique (noir et blanc)
- Le visage masqué

---

### 👁️ Étape 2 — Masque “intelligent” centré sur les yeux
Script: `scripts/etape_masque_repris.py`

Objectif:
- Détecter les yeux dans la région du visage (ROI).
- Calculer le centre et la distance entre les yeux.
- Créer un masque elliptique ajusté dynamiquement à la position et taille du visage.
- Appliquer un flou pour adoucir les bords du masque.

Exécution:
```bash
python scripts/etape_masque_repris.py
```

Résultat attendu (affichage côte à côte):
- Visage original
- Masque intelligent (flouté)
- Visage masqué

---

### 🔄 Étape 3 — Face Swap (Échange de visages)
Script: `scripts/face_swap.py`

Objectif:
- Détecter deux visages dans la même image.
- Extraire les régions correspondantes.
- Redimensionner les visages pour les adapter à la taille de l’autre.
- Appliquer un masque flouté (feathered mask) pour adoucir la fusion.
- Échanger les visages et afficher le résultat final.

Exécution:
```bash
python scripts/face_swap.py
```

Résultat attendu (fenêtre):
- À gauche: image originale
- À droite: image modifiée avec les visages échangés

---

## Conseils et dépannage

- L’image utilisée pour le face swap doit contenir exactement deux visages visibles et bien éclairés.
- Si une erreur apparaît du type:
  ```
  can't open/read file: check file path/integrity
  ```
  → Vérifiez simplement que le chemin de l’image est correct et qu’elle existe dans le dossier `images/`.

- Si les visages n’ont pas la même taille ou orientation, le résultat peut être un peu déformé. C’est normal pour une version “basique”.
- Ce TP est une introduction au face swap — les approches plus avancées utilisent des points de repère (landmarks) et un recalage géométrique plus précis.

---

## Ressources utiles

- Documentation OpenCV: https://docs.opencv.org/4.x/
- Classificateurs Haar Cascade officiels: https://github.com/opencv/opencv/tree/master/data/haarcascades
- Explication du face swap (tutoriel): https://learnopencv.com/face-swap-using-opencv-c-python/

---

## Environnement Python (optionnel mais recommandé)

Créer et activer un environnement virtuel:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
```

Installer les dépendances:
```bash
pip install -U pip
pip install opencv-python numpy
```
