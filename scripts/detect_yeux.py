#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 - Étape 2 : Détection des yeux dans les visages
Objectif : Pour chaque visage, détecter les yeux
"""

import cv2 as cv
import sys
import os

# Chemin du dossier contenant ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Dossier du script : {BASE_DIR}")

# Classificateurs Haar Cascade
face_cascade = cv.CascadeClassifier(
    os.path.join(BASE_DIR, '..', 'haarcascade_frontalface_default.xml')
)
eye_cascade = cv.CascadeClassifier(
    os.path.join(BASE_DIR, '..', 'haarcascade_eye.xml')
)

# Vérification
if face_cascade.empty():
    sys.exit("ERREUR : Impossible de charger haarcascade_frontalface_default.xml")
if eye_cascade.empty():
    sys.exit("ERREUR : Impossible de charger haarcascade_eye.xml")

# Charger l'image 'obama.jpg' depuis le dossier images
img_path = os.path.join(BASE_DIR, '..', 'images', 'obama.jpg')
img = cv.imread(img_path)

if img is None:
    sys.exit(f"ERREUR : Image introuvable : {img_path}")

print(f"✓ Image chargée : {img.shape[1]}x{img.shape[0]} pixels")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Détection des visages
faces = face_cascade.detectMultiScale(gray, 1.1, 8)
print(f"Nombre de visages détectés : {len(faces)}")

# Détection des yeux pour chaque visage
for i, (x, y, w, h) in enumerate(faces):
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    print(f"  → Visage {i} : {len(eyes)} yeux détectés")

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

# Affichage
cv.imshow('Visages et yeux', img)
cv.waitKey(0)
cv.destroyAllWindows()

print("✓ Détection terminée !")
