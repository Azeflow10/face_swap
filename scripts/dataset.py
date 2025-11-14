import os
import sys
import numpy as np
from PIL import Image
import cv2

# Haar pour detecter les visages
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def _extract_face(filepath):
    """
    Charge une image, détecte le visage, et retourne la région du visage.
    Retourne un tableau numpy.
    """

    # Chargement
    img_pil = Image.open(filepath).convert("L") 
    img = np.array(img_pil)

    # Détection
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError(f"Aucun visage détecté dans : {filepath}")
    if len(faces) > 1:
        raise ValueError(f"Plusieurs visages détectés dans : {filepath}")

    x, y, w, h = faces[0]

    # Extraction de la région du visage
    face_region = img[y:y+h, x:x+w]

    return face_region



def load_data(dataset_path):
    """
    Parcourt le dataset, sépare train/test, et retourne :
    (X_train, y_train, filenames_train), (X_test, y_test)
    """

    files = os.listdir(dataset_path)

    X_train = []
    y_train = []
    filenames_train = {}

    X_test = []
    y_test = []

    for file in files:
        if not "." in file:
            continue

        filepath = os.path.join(dataset_path, file)
        name = file.split(".")[0]        # subject03
        subject_id = int(name.replace("subject", "")) 

        try:
            face = _extract_face(filepath)
        except Exception as e:
            print("Erreur :", e)
            continue

        if file.endswith(".wink"):
            # Image de test
            X_test.append(face)
            y_test.append(subject_id)
        else:
            # Image d'entraînement
            X_train.append(face)
            y_train.append(subject_id)
            filenames_train[subject_id] = file

    if len(X_train) == 0 or len(X_test) == 0:
        print("Erreur : dataset vide ou mal structuré.")
        sys.exit()

    return (X_train, y_train, filenames_train), (X_test, y_test)
