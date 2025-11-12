#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 - Étape 1 : Détection de visages
Objectif : Détecter et encadrer tous les visages d'une image
"""

import cv2 as cv
import sys
import os

# Chemin du dossier contenant ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Dossier du script : {BASE_DIR}")

# Classificateur Haar Cascade pour les visages
face_cascade = cv.CascadeClassifier(
    os.path.join(BASE_DIR, '..', 'haarcascade_frontalface_default.xml')
)

if face_cascade.empty():
    sys.exit("ERREUR : Impossible de charger haarcascade_frontalface_default.xml")

# Charger l'image 'obama.jpg' depuis le dossier images
img_path = os.path.join(BASE_DIR, '..', 'images', 'obama.jpg')
img = cv.imread(img_path)

if img is None:
    sys.exit(f"ERREUR : Impossible de charger l'image : {img_path}")

print(f"✓ Image chargée : {img.shape[1]}x{img.shape[0]} pixels")

# Conversion en niveaux de gris
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Détection des visages
faces = face_cascade.detectMultiScale(gray, 1.1, 8)
print(f"Nombre de visages détectés : {len(faces)}")

# Dessiner un rectangle vert autour de chaque visage
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Afficher l'image
cv.imshow('Visages détectés', img)
cv.waitKey(0)
cv.destroyAllWindows()

print("✓ Détection terminée !")