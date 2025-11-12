#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 - Étape 3 : Calcul du centre des yeux
Objectif : Calculer et afficher le centre précis de chaque œil
"""

import cv2 as cv
import sys
import os

# ==========================================
# CHARGEMENT ET DÉTECTION
# ==========================================

print(f"Dossier de travail : {os.getcwd()}")

face_cascade = cv.CascadeClassifier('../haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('../haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    sys.exit("ERREUR : Classificateurs non chargés")

img = cv.imread('../images/obama.jpg')

if img is None:
    sys.exit("ERREUR : Image non trouvée")

print(f"✓ Image chargée : {img.shape[1]}x{img.shape[0]} pixels")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 8)

print(f"Nombre de visages : {len(faces)}")

# ==========================================
# TRAITEMENT DE CHAQUE VISAGE
# ==========================================

for i, (x, y, w, h) in enumerate(faces):
    
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    
    eyes_centers = []
    
    for (ex, ey, ew, eh) in eyes:
        
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        
        # TODO 3.1 : Calcul du centre de l'œil
        center_x = ex + ew // 2
        center_y = ey + eh // 2
        
        # TODO 3.2 : Dessiner un cercle rouge au centre
        cv.circle(roi_color, (center_x, center_y), 3, (0, 0, 255), -1)
        
        eyes_centers.append((center_x, center_y))
    
    print(f"Visage {i} - Centres des yeux : {eyes_centers}")

# ==========================================
# AFFICHAGE
# ==========================================

cv.imshow('Centres des yeux', img)
cv.waitKey(0)
cv.destroyAllWindows()

print("✓ Détection terminée !")
