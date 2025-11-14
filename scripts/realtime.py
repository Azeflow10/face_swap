import cv2
import numpy as np
import os
from dataset import load_data

# Définir le chemin du dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../yalefaces")

# Chargement du dataset
(train_X, train_y, filenames_train), (test_X, test_y) = load_data(DATASET_DIR)

# Uniformisation des tailles
target_h, target_w = train_X[0].shape
train_X_resized = [cv2.resize(f, (target_w, target_h)) for f in train_X]

# Creation du modèle LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(train_X_resized, np.array(train_y))

# Haar cascade pour la edtection des visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ouverture de la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        face_region = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (target_w, target_h))

        # Prédiction
        predicted_id, confidence = face_recognizer.predict(face_resized)

        # Récupération du nom du fichier correspondant
        label = filenames_train.get(predicted_id, "Inconnu")

        # Dessiner rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Afficher le nom au-dessus du rectangle
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Reconnaissance faciale", frame)

    # Sortie avec la lettre q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermeture propre de la webcam et des fenetre
cap.release()
cv2.destroyAllWindows()
