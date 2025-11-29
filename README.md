# 🔄 Face Swap - Détection et Manipulation de Visages

Projet de détection et manipulation de visages utilisant OpenCV et les classificateurs Haar Cascade pour la reconnaissance faciale et oculaire.

## 📋 Description

Ce projet permet de détecter des visages et des yeux dans des images en utilisant les algorithmes de détection d'OpenCV.  Il inclut des scripts pour différentes fonctionnalités de détection et un système de vérification d'environnement.

## 🚀 Fonctionnalités

- **Détection de visages** : Utilise Haar Cascade pour identifier les visages dans les images
- **Détection des yeux** : Localise les yeux dans les visages détectés
- **Détection des centres** : Calcule les points centraux des éléments détectés
- **Vérification d'environnement** : Script automatique pour valider la configuration

## 📁 Structure du Projet

```
face_swap/
├── scripts/
│   ├── detect_visage.py      # Détection de visages
│   ├── detect_yeux.py         # Détection des yeux
│   └── detect_centres.py      # Détection des centres
├── images/                     # Dossier pour les images de test
├── haarcascade_frontalface_default.xml  # Modèle de détection de visages
├── haarcascade_eye.xml        # Modèle de détection des yeux
├── verifier-env.py            # Script de vérification de l'environnement
└── README.md
```

## 🛠️ Prérequis

- Python 3.x
- OpenCV (cv2)

## 📦 Installation

1. Clonez le repository :
```bash
git clone https://github.com/Azeflow10/face_swap.git
cd face_swap
```

2. Installez les dépendances :
```bash
pip install opencv-python
```

3. Vérifiez votre environnement :
```bash
python verifier-env.py
```

## 💻 Utilisation

### Vérification de l'environnement

Avant de commencer, exécutez le script de vérification pour vous assurer que tous les fichiers nécessaires sont présents :

```bash
python verifier-env.py
```

Ce script vérifie :
- ✓ La présence des fichiers XML (classificateurs Haar Cascade)
- ✓ L'existence du dossier images
- ✓ La capacité à charger les images
- ✓ La validité des classificateurs

### Détection de visages

```bash
python scripts/detect_visage.py
```

### Détection des yeux

```bash
python scripts/detect_yeux.py
```

### Détection des centres

```bash
python scripts/detect_centres.py
```

## 📊 Fichiers de Modèles

Le projet utilise les classificateurs Haar Cascade d'OpenCV :

- **haarcascade_frontalface_default.xml** : Modèle pré-entraîné pour la détection de visages de face
- **haarcascade_eye.xml** : Modèle pré-entraîné pour la détection des yeux

## 🖼️ Images

Placez vos images de test dans le dossier `images/`.  Les formats supportés sont :
- `. jpg` / `.jpeg`
- `.png`

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3.  Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est un projet éducatif de démonstration de détection faciale avec OpenCV. 

## 👤 Auteur

**Azeflow10**
- GitHub: [@Azeflow10](https://github.com/Azeflow10)

## 🙏 Remerciements

- OpenCV pour les outils de vision par ordinateur
- La communauté OpenCV pour les classificateurs Haar Cascade pré-entraînés

---

⭐ N'oubliez pas de mettre une étoile si ce projet vous a été utile ! 
```
