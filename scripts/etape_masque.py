#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
face_cascade = cv2.CascadeClassifier(os.path.join(ROOT, "haarcascade_frontalface_default.xml"))
img_path = os.path.join(ROOT, "images", "obama.jpg")
img = cv2.imread(img_path)
if img is None:
    raise SystemExit(f"Impossible de charger l'image : {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

if len(faces) == 0:
    print("Aucun visage détecté.")
else:
    x,y,w,h = faces[0]  
    face = img[y:y+h, x:x+w].copy()
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # création du masque gris (single channel)
    mask = np.zeros((h, w), dtype=np.uint8)

    # ellipse centrée
    center = (w//2, h//2)
    axes = (int(w/2.2), int(h/2.4))  # ajuster pour éviter bords trop proches
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # appliquer le masque
    face_masked = cv2.bitwise_and(face, face, mask=mask)

    # pour visualisation côte-à-côte
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack([face, mask_color, face_masked])
    cv2.imshow("Face | Mask | Face masked", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
