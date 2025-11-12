#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de vérification pour le TP1
À exécuter avant de commencer pour vérifier que tout est en place
"""

import cv2 as cv
import os

print("=" * 50)
print("VÉRIFICATION DE L'ENVIRONNEMENT TP1")
print("=" * 50)

# 1. Dossier de travail
print(f"\n📁 Dossier de travail : {os.getcwd()}")

# 2. Vérifier les fichiers XML
fichiers_xml = ['haarcascade_frontalface_default.xml', 'haarcascade_eye.xml']
print("\n📄 Fichiers XML :")
for fichier in fichiers_xml:
    if os.path.exists(fichier):
        cascade = cv.CascadeClassifier(fichier)
        if cascade.empty():
            print(f"  ✗ {fichier} existe mais est invalide")
        else:
            print(f"  ✓ {fichier}")
    else:
        print(f"  ✗ {fichier} MANQUANT")

# 3. Vérifier le dossier images
print("\n🖼️  Dossier images :")
if os.path.exists('images'):
    print(f"  ✓ Le dossier 'images' existe")
    images = [f for f in os.listdir('images') if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  → {len(images)} image(s) trouvée(s) :")
    for img in images:
        print(f"     - {img}")
else:
    print(f"  ✗ Le dossier 'images' n'existe pas")

# 4. Test de chargement
print("\n🧪 Test de chargement d'image :")
if os.path.exists('images') and images:
    test_img = f'images/{images[0]}'
    img = cv.imread(test_img)
    if img is None:
        print(f"  ✗ Impossible de charger {test_img}")
    else:
        print(f"  ✓ Image chargée avec succès : {img.shape[1]}x{img.shape[0]} pixels")
else:
    print(f"  ⚠ Aucune image à tester")

print("\n" + "=" * 50)
print("Vérification terminée !")
print("=" * 50)