import cv2
import numpy as np

# Chargement des classificateurs pour le visage et les yeux
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Chargement de l'image
image = cv2.imread('images/brad-angelina.jpg')  # à adapter selon ton image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détection des visages
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
print(f"{len(faces)} visages détectés.")

if len(faces) == 0:
    exit("Aucun visage détecté.")

# Boucle sur chaque visage détecté
for (x, y, w, h) in faces:
    face_roi = image[y:y+h, x:x+w]
    gray_face = gray[y:y+h, x:x+w]

    # Détection des yeux dans la région du visage
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 4)
    print(f"Nombre d'yeux détectés : {len(eyes)}")

    # Création d'un masque noir
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(eyes) >= 2:
        # Tri des yeux par position horizontale
        eyes = sorted(eyes, key=lambda e: e[0])

        # Coordonnées des deux yeux
        (x1, y1, w1, h1) = eyes[0]
        (x2, y2, w2, h2) = eyes[1]

        # Calcul des centres des yeux
        eye1_center = (x1 + w1 // 2, y1 + h1 // 2)
        eye2_center = (x2 + w2 // 2, y2 + h2 // 2)

        # Centre entre les deux yeux et distance entre eux
        center_x = int((eye1_center[0] + eye2_center[0]) / 2)
        center_y = int((eye1_center[1] + eye2_center[1]) / 2)
        eye_distance = abs(eye2_center[0] - eye1_center[0])

        # Définition de la taille et du décalage du masque
        radius_x = int(eye_distance * 1.3)
        radius_y = int(eye_distance * 1.6)
        center_y = int(center_y + eye_distance * 0.4)

        # Dessin de l'ellipse blanche sur le masque
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)

    else:
        # Masque par défaut si les yeux ne sont pas détectés
        center = (w // 2, h // 2)
        axes = (int(w / 2.2), int(h / 2.4))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Adoucissement des bords avec un flou gaussien
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)

    # Application du masque sur le visage
    masked_face = cv2.bitwise_and(face_roi, face_roi, mask=mask_blur)

    # Affichage du résultat
    combined = np.hstack((face_roi, cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR), masked_face))
    cv2.imshow("Masque intelligent centré sur les yeux", combined)
    cv2.waitKey(0)

cv2.destroyAllWindows()
