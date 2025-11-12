import cv2
import numpy as np

# Chargement du classificateur pour la détection de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Chargement de l'image contenant deux visages
image = cv2.imread('images/obama.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détection des visages
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"{len(faces)} visages détectés.")

# Vérification qu'on a bien au moins deux visages
if len(faces) < 2:
    print("Erreur : il faut au moins deux visages dans l'image.")
    exit()

# Extraction des deux premiers visages détectés
(x1, y1, w1, h1) = faces[0]
(x2, y2, w2, h2) = faces[1]

face1 = image[y1:y1+h1, x1:x1+w1]
face2 = image[y2:y2+h2, x2:x2+w2]

# Redimensionnement croisé : chaque visage prend la taille de l'autre
face1_resized = cv2.resize(face1, (w2, h2))
face2_resized = cv2.resize(face2, (w1, h1))

# Fonction pour créer un masque flouté (feathered mask)
def create_feathered_mask(width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (int(width * 0.35), int(height * 0.40))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return mask

# Création des masques pour chaque visage
mask1 = create_feathered_mask(w2, h2)
mask2 = create_feathered_mask(w1, h1)

# Conversion en float pour les calculs
mask1_f = mask1.astype(np.float32) / 255.0
mask2_f = mask2.astype(np.float32) / 255.0
mask1_3 = cv2.merge([mask1_f, mask1_f, mask1_f])
mask2_3 = cv2.merge([mask2_f, mask2_f, mask2_f])

face1_f = face1_resized.astype(np.float32)
face2_f = face2_resized.astype(np.float32)

# Vérification et harmonisation des tailles
h1, w1_, _ = face1_f.shape
h2, w2_, _ = face2_f.shape
if (h1, w1_) != (h2, w2_):
    # On redimensionne tout à la taille la plus petite pour éviter les erreurs
    min_h, min_w = min(h1, h2), min(w1_, w2_)
    face1_f = cv2.resize(face1_f, (min_w, min_h))
    face2_f = cv2.resize(face2_f, (min_w, min_h))
    mask1_3 = cv2.resize(mask1_3, (min_w, min_h))
    mask2_3 = cv2.resize(mask2_3, (min_w, min_h))

# Mélange progressif (blending) des visages
swap1 = (face1_f * mask1_3 + face2_f * (1 - mask1_3)).astype(np.uint8)
swap2 = (face2_f * mask2_3 + face1_f * (1 - mask2_3)).astype(np.uint8)

# Copie de l'image originale pour y insérer les visages échangés
image_swapped = image.copy()

# Insertion dans les zones correspondantes
image_swapped[y1:y1+swap2.shape[0], x1:x1+swap2.shape[1]] = swap2
image_swapped[y2:y2+swap1.shape[0], x2:x2+swap1.shape[1]] = swap1

# Affichage côte à côte : image originale et image modifiée
compare = np.hstack((image, image_swapped))

cv2.imshow("Face Swap - Original | Modifie", compare)
cv2.waitKey(0)
cv2.destroyAllWindows()
