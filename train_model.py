print("Training script started...")
import os
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestRegressor
import pickle

# Paths
IMG_DIR = "SCUT-FBP5500_v2/Images"
LABEL_FILE = "SCUT-FBP5500_v2/All_Ratings.xlsx"
MODEL_FILE = "model.pkl"

import os
print("Images folder exists:", os.path.exists(IMG_DIR))
print("Ratings file exists:", os.path.exists(LABEL_FILE))

if not os.path.exists(IMG_DIR) or not os.path.exists(LABEL_FILE):
    raise FileNotFoundError("Dataset paths are incorrect!")

# Load the Excel file
df = pd.read_excel(LABEL_FILE)

# Select only the needed columns
df = df[['Filename', 'Rating']]
df.columns = ['image', 'rating']

# Randomly pick 1000 images for faster training
df = df.sample(n=1000, random_state=42).reset_index(drop=True)



# Mediapipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# Prepare dataset
X, y = [], []
for _, row in df.iterrows():
    path = os.path.join(IMG_DIR, row["image"])
    features = extract_features(path)
    if features:
        X.append(features)
        y.append(row["rating"])

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples.")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_FILE}")
