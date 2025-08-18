#!/usr/bin/env python3
"""
Updated FaceAnalyzer backend using MediaPipe FaceMesh for real measurements.
Drop this in place of your old app_final_debug.py
"""

from flask import Flask, render_template, request, jsonify
import os
import base64
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import mediapipe as mp
import math
import io
import pickle  # === ML MODEL ADDITIONS ===

# === MediaPipe FaceMesh setup ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)  # initialize once here

# Load trained ML model at app startup
with open("model.pkl", "rb") as f:
    ml_model = pickle.load(f)

app = Flask(__name__)
app.secret_key = 'facial_analysis_secret_key_2024'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

mp_face = mp.solutions.face_mesh

# === ML MODEL ADDITIONS: load trained model (if present) ===
MODEL_FILE = "model.pkl"
ml_model = None
try:
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            ml_model = pickle.load(f)
        print(f"✅ Loaded ML model from {MODEL_FILE}")
    else:
        print(f"⚠️ ML model file not found at {MODEL_FILE}. 'ml_predicted_rating' will be unavailable.")
except Exception as e:
    print(f"⚠️ Failed to load ML model: {e}")
    ml_model = None

def extract_features_for_model(img_path):
    """
    Extract features EXACTLY like your train_model.py:
    - read image
    - run MediaPipe FaceMesh (static_image_mode=True)
    - take the FIRST face's 468 landmarks
    - flatten [x, y, z] for each -> 1404-dim vector
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    coords = []  # make sure this is defined
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    return coords  # this will now always be defined if a face is found


# === ML MODEL ADDITIONS ===
def get_ml_rating(image_path):
    features = extract_features_for_model(image_path)  # use same function as training
    if features:
        rating = ml_model.predict([features])[0]
        return round(float(rating), 1)  # e.g., 6.8
    return 0.0

    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use a short-lived FaceMesh instance (matches training usage style)
    with mp_face.FaceMesh(static_image_mode=True) as fm:
        results = fm.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords
# === END ML MODEL ADDITIONS ===


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ratio_score(ratio, target, tol=0.25):
    """Map a ratio to 0-100 (tolerant). tol is fraction of target that maps to zero."""
    if target <= 0:
        return 0.0
    diff = abs(ratio - target) / target
    val = max(0.0, 100.0 * (1.0 - diff / tol))
    return float(round(min(100.0, val), 2))

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bbox_from_landmarks(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x, min_y, max_x, max_y)

def rotate_image(image, angle, center=None):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M

def transform_points(points, M):
    pts = np.array(points)
    ones = np.ones((pts.shape[0], 1))
    pts_hom = np.hstack([pts, ones])
    transformed = pts_hom.dot(M.T)
    return [(int(x), int(y)) for x,y in transformed]

class FacialAnalyzer:
    def __init__(self):
        # indices sets for common facial regions (MediaPipe landmark indices)
        self.left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        self.right_eye_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362]
        self.mouth_left = 61
        self.mouth_right = 291
        self.nose_tip = 1
        self.chin = 152
        self.left_jaw = 234
        self.right_jaw = 454

    def get_landmarks(self, image_bgr, max_faces=2):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=max_faces, refine_landmarks=True) as fm:
            results = fm.process(image_rgb)
            if not results.multi_face_landmarks:
                return []
            h, w = image_bgr.shape[:2]
            faces = []
            for face_landmarks in results.multi_face_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                faces.append(pts)
            return faces

    def pick_primary_face(self, faces):
        # faces: list of landmark lists -> choose by bbox area
        best = None
        best_area = 0
        for pts in faces:
            x1,y1,x2,y2 = bbox_from_landmarks(pts)
            area = (x2-x1) * (y2-y1)
            if area > best_area:
                best_area = area
                best = pts
        return best

    def align_and_crop_face(self, image, pts, output_size=400, expand=0.25):
        # compute eye centers
        le = np.mean([pts[i] for i in self.left_eye_idx if i < len(pts)], axis=0)
        re = np.mean([pts[i] for i in self.right_eye_idx if i < len(pts)], axis=0)
        # angle to horizontal
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        angle = math.degrees(math.atan2(dy, dx))
        # rotate image around center between eyes
        eye_center = ((le[0]+re[0])//2, (le[1]+re[1])//2)
        rotated, M = rotate_image(image, angle, center=eye_center)
        # map landmarks
        new_pts = transform_points(pts, M)
        x1,y1,x2,y2 = bbox_from_landmarks(new_pts)
        # expand bbox
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * expand)
        pad_y = int(h * expand)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(rotated.shape[1], x2 + pad_x)
        y2 = min(rotated.shape[0], y2 + pad_y)
        crop = rotated[y1:y2, x1:x2].copy()
        # also transform new_pts to crop coords
        cropped_pts = [(p[0] - x1, p[1] - y1) for p in new_pts]
        # resize to fixed
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None, None, None
        face_resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
        # compute scale to map cropped_pts -> resized coords
        scale_x = output_size / (x2 - x1)
        scale_y = output_size / (y2 - y1)
        resized_pts = [(int(p[0]*scale_x), int(p[1]*scale_y)) for p in cropped_pts]
        return face_resized, resized_pts, (x1,y1,x2,y2)

    def compute_symmetry(self, face_crop):
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            h,w = gray.shape
            half = w // 2
            left = gray[:, :half]
            right = cv2.flip(gray[:, w-half:], 1)
            minw = min(left.shape[1], right.shape[1])
            left = left[:, :minw]
            right = right[:, :minw]
            diff = cv2.absdiff(left, right)
            score = 100.0 - (diff.mean() / 255.0) * 100.0
            score = float(round(max(0.0, min(100.0, score)), 2))
            return score, diff
        except Exception:
            return 50.0, None

    def compute_skin_scores(self, face_crop):
        # inner-face crop to avoid hair/background
        h,w = face_crop.shape[:2]
        cx1, cy1, cx2, cy2 = int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.75)
        inner = face_crop[cy1:cy2, cx1:cx2]
        if inner.size == 0:
            return {"smoothness": 50.0, "uniformity": 50.0, "overall": 50.0, "blemish_pct": 0.0}
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        high_freq = cv2.absdiff(gray, blur)
        smoothness = max(0.0, 100.0 - (high_freq.mean()/255.0)*100.0)
        # uniformity using L channel of LAB
        lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB)
        L = lab[:,:,0].astype(np.float32)
        uniformity = max(0.0, 100.0 - (L.std()/255.0)*100.0)
        # blemish: simple redness/spot detection in HSV (crude)
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        # look for high saturation + hue in red/orange range (acne/redness)
        low1 = np.array([0, 40, 40])
        high1 = np.array([15, 255, 255])
        low2 = np.array([170, 40, 40])
        high2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, low1, high1)
        mask2 = cv2.inRange(hsv, low2, high2)
        mask = cv2.bitwise_or(mask1, mask2)
        blemish_pct = (mask > 0).sum() / (mask.size) * 100.0
        # combine: penalize blemish area slightly
        overall = max(0.0, (0.6 * smoothness + 0.4 * uniformity) - (blemish_pct * 0.2))
        return {
            "smoothness": float(round(smoothness,2)),
            "uniformity": float(round(uniformity,2)),
            "overall": float(round(max(0.0, min(100.0, overall)),2)),
            "blemish_pct": float(round(blemish_pct,2))
        }

    def analyze_face(self, image_path):
        import time
        start = time.time()
        print("🟢 [1] Starting analysis for", image_path)

        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not load image"}

        orig_h, orig_w = img.shape[:2]
        faces = self.get_landmarks(img, max_faces=3)
        if not faces:
            return {"error": "No face detected in the image", "face_detected": False}

        primary = self.pick_primary_face(faces)
        if primary is None:
            return {"error": "No face chosen", "face_detected": False}

        face_crop, resized_pts, bbox = self.align_and_crop_face(img, primary, output_size=400)
        if face_crop is None:
            return {"error": "Failed to align/crop face", "face_detected": False}

        # Save detected face image (annotated)
        annotated = img.copy()
        x1,y1,x2,y2 = bbox
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,0), 2)
        detected_path = os.path.join(STATIC_FOLDER, "detected_face.jpg")
        cv2.imwrite(detected_path, annotated)

        # compute symmetry
        symmetry_score, symmetry_diff = self.compute_symmetry(face_crop)
        if symmetry_diff is not None:
            cv2.imwrite(os.path.join(STATIC_FOLDER, "symmetry_diff.jpg"), symmetry_diff)

        # landmarks in resized crop coords (resized_pts)
        pts = resized_pts
        # compute many distances and ratios
        # eye centers
        left_eye_pts = [pts[i] for i in self.left_eye_idx if i < len(pts)]
        right_eye_pts = [pts[i] for i in self.right_eye_idx if i < len(pts)]
        if left_eye_pts and right_eye_pts:
            left_eye_center = tuple(np.mean(left_eye_pts, axis=0).astype(int))
            right_eye_center = tuple(np.mean(right_eye_pts, axis=0).astype(int))
        else:
            # fallback to approximate
            left_eye_center = (int(400*0.3), int(400*0.35))
            right_eye_center = (int(400*0.7), int(400*0.35))

        interocular = distance(left_eye_center, right_eye_center)
        face_h = max(p[1] for p in pts) - min(p[1] for p in pts)
        face_w = max(p[0] for p in pts) - min(p[0] for p in pts)
        if face_w == 0: face_w = 1
        if face_h == 0: face_h = 1

        # nose tip and chin
        nose_pt = pts[self.nose_tip] if self.nose_tip < len(pts) else (200, 180)
        chin_pt = pts[self.chin] if self.chin < len(pts) else (200, 380)
        nose_length = distance(nose_pt, chin_pt)

        # mouth corners
        mouth_l = pts[self.mouth_left] if self.mouth_left < len(pts) else (150, 300)
        mouth_r = pts[self.mouth_right] if self.mouth_right < len(pts) else (250, 300)
        mouth_width = distance(mouth_l, mouth_r)

        # jaw
        left_j = pts[self.left_jaw] if self.left_jaw < len(pts) else (120, 350)
        right_j = pts[self.right_jaw] if self.right_jaw < len(pts) else (280, 350)
        jaw_diff = abs(distance(chin_pt, left_j) - distance(chin_pt, right_j))

        # Ratio-based scores (targets are initial heuristics)
        r_interocular_facew = interocular / face_w
        r_faceh_facew = face_h / face_w
        r_nose_faceh = nose_length / face_h
        r_mouth_interocular = mouth_width / interocular if interocular>0 else 0.5

        interocular_score = ratio_score(r_interocular_facew, target=0.46, tol=0.3)
        golden_ratio_face_score = ratio_score(r_faceh_facew, target=1.6, tol=0.3)
        nose_score = ratio_score(r_nose_faceh, target=0.33, tol=0.3)
        mouth_score = ratio_score(r_mouth_interocular, target=0.65, tol=0.35)

        # eye size and symmetry
        left_eye_bbox = (min([p[0] for p in left_eye_pts]), min([p[1] for p in left_eye_pts]),
                         max([p[0] for p in left_eye_pts]), max([p[1] for p in left_eye_pts])) if left_eye_pts else (0,0,0,0)
        right_eye_bbox = (min([p[0] for p in right_eye_pts]), min([p[1] for p in right_eye_pts]),
                         max([p[0] for p in right_eye_pts]), max([p[1] for p in right_eye_pts])) if right_eye_pts else (0,0,0,0)
        left_eye_area = max(1, (left_eye_bbox[2]-left_eye_bbox[0])*(left_eye_bbox[3]-left_eye_bbox[1]))
        right_eye_area = max(1, (right_eye_bbox[2]-right_eye_bbox[0])*(right_eye_bbox[3]-right_eye_bbox[1]))
        eye_size_diff_pct = abs(left_eye_area - right_eye_area) / max(left_eye_area, right_eye_area)
        eye_size_score = float(round(max(0.0, 100.0 - eye_size_diff_pct*100.0),2))
        eye_spacing_score = interocular_score
        eye_symmetry_score = symmetry_score  # rough proxy

        # jaw symmetry/definition score (simple heuristic)
        jaw_symmetry_score = float(round(max(0.0, 100.0 - (jaw_diff / face_w) * 200.0), 2))
        jaw_definition_score = float(round(max(0.0, min(100.0, 50.0 + (face_w/float(face_h))*50.0)),2))

        # skin analysis
        skin = self.compute_skin_scores(face_crop)

        # combine golden ratio as weighted average of ratio measures
        golden_score = float(round((golden_ratio_face_score*0.4 + interocular_score*0.3 + nose_score*0.15 + mouth_score*0.15),2))

        # facial features structure
        facial_features = {
            "eyes": {
                "symmetry": float(round(eye_symmetry_score,2)),
                "count": 2,
                "spacing_score": float(round(eye_spacing_score,2)),
                "size_score": float(round(eye_size_score,2))
            },
            "nose": {
                "proportion_score": float(round(nose_score,2)),
                "symmetry_score": float(round(symmetry_score,2))
            },
            "mouth": {
                "proportion_score": float(round(mouth_score,2)),
                "symmetry_score": float(round(symmetry_score,2))
            },
            "jawline": {
                "definition_score": float(round(jaw_definition_score,2)),
                "symmetry_score": float(round(jaw_symmetry_score,2))
            }
        }

        # overall rating (initial weights)
        weights = {
            "symmetry": 0.25,
            "golden": 0.15,
            "skin": 0.20,
            "eyes": 0.20,
            "facial_harmony": 0.20
        }
        eyes_avg = np.mean([facial_features["eyes"]["symmetry"], facial_features["eyes"]["size_score"], facial_features["eyes"]["spacing_score"]])
        facial_harmony = np.mean([facial_features["nose"]["proportion_score"], facial_features["mouth"]["proportion_score"], facial_features["jawline"]["definition_score"]])

        overall = (
            weights["symmetry"] * symmetry_score +
            weights["golden"] * golden_score +
            weights["skin"] * skin["overall"] +
            weights["eyes"] * eyes_avg +
            weights["facial_harmony"] * facial_harmony
        )
        overall_score = float(round(max(0.0, min(100.0, overall)), 2))

        analysis = {
            "face_detected": True,
            "filename": os.path.basename(image_path),
            "upload_time": datetime.now().isoformat(),
            "overall_rating": overall_score,
            "symmetry_score": float(round(symmetry_score,2)),
            "golden_ratio_score": float(round(golden_score,2)),
            "skin_quality": {
                "smoothness": skin["smoothness"],
                "uniformity": skin["uniformity"],
                "overall": skin["overall"],
                "blemish_pct": skin["blemish_pct"]
            },
            "facial_features": facial_features,
            "detailed_breakdown": {
                "symmetry": {"explanation": "Pixel-level left-right symmetry measured from aligned face crop."},
                "proportions": {"explanation": "Ratios measured with MediaPipe landmarks compared to heuristic targets."},
                "skin_quality": {"explanation": "Smoothness and uniformity computed from L-channel and high-frequency texture."},
                "eyes": {"explanation": "Eye spacing and size measured using landmark clusters."}
            }
        }

        elapsed = round(time.time() - start, 2)
        print(f"✅ Analysis finished in {elapsed}s — overall {overall_score}/100")
        return analysis

facial_analyzer = FacialAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # analyze (your original heuristic analysis)
        analysis_result = facial_analyzer.analyze_face(filepath)

        # === ML MODEL ADDITIONS: predict with trained model ===
        analysis_result["ml_rating"] = get_ml_rating(filepath)
        try:
            if ml_model is not None:
                features = extract_features_for_model(filepath)
                if features:
                    pred = ml_model.predict([features])[0]
                    analysis_result['ml_predicted_rating'] = round(float(pred), 2)
                else:
                    analysis_result['ml_predicted_rating'] = "Face not detected by ML model"
            else:
                analysis_result['ml_predicted_rating'] = "Model not loaded"
        except Exception as e:
            analysis_result['ml_predicted_rating'] = f"Prediction error: {str(e)}"
        # === END ML MODEL ADDITIONS ===

        # add file size
        try:
            analysis_result['file_size'] = os.path.getsize(filepath)
        except Exception:
            analysis_result['file_size'] = None

        # attach image (base64) for front-end
        try:
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                # ensure proper MIME guess (we keep it jpeg for simplicity)
                analysis_result['image_data'] = f"data:image/jpeg;base64,{img_data}"
        except Exception as e:
            analysis_result['image_data'] = None

        # cleanup uploaded file to avoid storage growth (optional)
        try:
            os.remove(filepath)
        except Exception:
            pass

        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({'error': f'Failed: {str(e)}'}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("🚀 Starting FaceAnalyzer Pro Flask Application (MediaPipe version + ML)")
    print(f"🌐 Visit http://127.0.0.1:{port}/")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
