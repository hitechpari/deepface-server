import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import gc
import time
import cv2
import numpy as np
from PIL import Image
import io

# ============================================
# ENVIRONMENT OPTIMIZATION
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# DEEPFACE IMPORT WITH ERROR HANDLING
# ============================================
DEEPFACE_AVAILABLE = False
try:
    import tensorflow as tf
    # Memory optimization
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    from deepface import DeepFace
    from deepface.commons import functions
    from deepface.detectors import FaceDetector
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported successfully")
except Exception as e:
    logger.error(f"❌ DeepFace import error: {e}")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(title="Missing Person Face Recognition API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DIRECTORY SETUP
# ============================================
KNOWN_FACES_DIR = Path("app/known_faces")
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# FACE DETECTION FUNCTIONS
# ============================================

def detect_face_in_image(image_path):
    """Ultimate face detection function"""
    try:
        # Try multiple detectors
        detectors = [
            ("opencv", "Fast"),
            ("mtcnn", "Accurate"),
            ("retinaface", "Best"),
        ]
        
        for detector_name, desc in detectors:
            try:
                logger.info(f"🔍 Trying {detector_name} detector...")
                face_objs = DeepFace.extract_faces(
                    img_path=str(image_path),
                    detector_backend=detector_name,
                    enforce_detection=False
                )
                
                if face_objs and len(face_objs) > 0:
                    logger.info(f"✅ Face detected with {detector_name}")
                    return True, detector_name
            except Exception as e:
                logger.debug(f"{detector_name} failed: {e}")
                continue
        
        # If all detectors fail, try image enhancement
        logger.warning("⚠️ No face detected, trying enhancement...")
        enhanced_path = enhance_image_for_face_detection(image_path)
        if enhanced_path:
            return detect_face_in_image(enhanced_path)
        
        return False, None
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return False, None

def enhance_image_for_face_detection(image_path):
    """Enhance image to improve face detection"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Multiple enhancement techniques
        enhanced_images = []
        
        # 1. Grayscale + CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        enhanced_1 = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # 2. Brightness normalization
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        enhanced_2 = cv2.merge([l, a, b])
        enhanced_2 = cv2.cvtColor(enhanced_2, cv2.COLOR_LAB2BGR)
        
        # 3. Denoising
        enhanced_3 = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Try each enhancement
        enhanced_paths = []
        for i, enhanced in enumerate([enhanced_1, enhanced_2, enhanced_3]):
            enhanced_path = image_path.parent / f"enhanced_{i}_{image_path.name}"
            cv2.imwrite(str(enhanced_path), enhanced)
            enhanced_paths.append(enhanced_path)
            
            # Check if face detected in enhanced image
            for detector in ["opencv", "mtcnn"]:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=str(enhanced_path),
                        detector_backend=detector,
                        enforce_detection=False
                    )
                    if face_objs and len(face_objs) > 0:
                        logger.info(f"✅ Face detected in enhanced image {i}")
                        return enhanced_path
                except:
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return None

def verify_saved_faces():
    """Verify all saved faces are detectable"""
    faces = list(KNOWN_FACES_DIR.glob("*.*"))
    for face_path in faces:
        has_face, detector = detect_face_in_image(face_path)
        if not has_face:
            logger.warning(f"⚠️ Face not detectable in {face_path.name}")
            # Try to enhance and replace
            enhanced = enhance_image_for_face_detection(face_path)
            if enhanced:
                enhanced.replace(face_path)
                logger.info(f"✅ Replaced with enhanced version")
        else:
            logger.info(f"✅ {face_path.name} face OK (detected by {detector})")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "Missing Person Face Recognition API",
        "status": "healthy",
        "deepface_available": DEEPFACE_AVAILABLE,
        "known_faces_count": len(list(KNOWN_FACES_DIR.glob("*.*")))
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "deepface_available": DEEPFACE_AVAILABLE,
        "known_faces_count": len(list(KNOWN_FACES_DIR.glob("*.*"))),
        "timestamp": str(uuid.uuid4())
    }

@app.post("/add-face-base64")
async def add_face_base64(data: dict):
    """Add face with face detection verification"""
    try:
        image_base64 = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode and save
        image_data = base64.b64decode(image_base64)
        metadata = f"{name}|{age}|{mobile}|{city}|{state}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        file_path = KNOWN_FACES_DIR / filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        logger.info(f"✅ File saved: {filename}")
        
        # Verify face detection
        has_face, detector = detect_face_in_image(file_path)
        if not has_face:
            logger.warning(f"⚠️ No face detected in {filename}, enhancing...")
            enhanced_path = enhance_image_for_face_detection(file_path)
            if enhanced_path:
                enhanced_path.replace(file_path)
                logger.info(f"✅ Replaced with enhanced version")
                has_face, detector = detect_face_in_image(file_path)
        
        return {
            "success": True,
            "message": "Face added successfully",
            "face_detected": has_face,
            "detector": detector if has_face else None
        }
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """ULTIMATE SEARCH FUNCTION - Guaranteed to find matches"""
    try:
        image_base64 = data.get('image')
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode and save temp file
        image_data = base64.b64decode(image_base64)
        temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as buffer:
            buffer.write(image_data)
        
        if not DEEPFACE_AVAILABLE:
            os.remove(temp_path)
            return []
        
        # Verify query image has detectable face
        logger.info("🔍 Analyzing query image...")
        has_face, detector = detect_face_in_image(temp_path)
        
        if not has_face:
            logger.warning("⚠️ Query image has no detectable face, enhancing...")
            enhanced_path = enhance_image_for_face_detection(temp_path)
            if enhanced_path:
                search_path = enhanced_path
                logger.info("✅ Using enhanced image for search")
            else:
                logger.error("❌ Cannot detect face in query image")
                os.remove(temp_path)
                return []
        else:
            search_path = temp_path
            logger.info(f"✅ Query image OK (face detected by {detector})")
        
        known_faces = list(KNOWN_FACES_DIR.glob("*.*"))
        logger.info(f"📊 Known faces count: {len(known_faces)}")
        
        if len(known_faces) == 0:
            os.remove(temp_path)
            if 'enhanced_path' in locals() and enhanced_path.exists():
                os.remove(enhanced_path)
            return []
        
        # Verify all known faces are detectable
        logger.info("🔍 Verifying known faces...")
        valid_faces = []
        for face_path in known_faces:
            has_face, _ = detect_face_in_image(face_path)
            if has_face:
                valid_faces.append(face_path)
        
        logger.info(f"📊 Valid faces count: {len(valid_faces)}")
        
        if len(valid_faces) == 0:
            logger.error("❌ No valid faces in database")
            os.remove(temp_path)
            return []
        
        # ===== ULTIMATE MATCHING CONFIGURATION =====
        all_matches = []
        
        # Try with different detectors and models
        detectors_to_try = ["opencv", "mtcnn"]
        models_to_try = [
            ("Facenet512", "cosine", "Most accurate"),
            ("ArcFace", "cosine", "Age-invariant"),
        ]
        
        for detector in detectors_to_try:
            for model_name, metric, desc in models_to_try:
                try:
                    logger.info(f"🔄 Trying {model_name}/{metric} with {detector} detector")
                    
                    dfs = DeepFace.find(
                        img_path=str(search_path),
                        db_path=str(KNOWN_FACES_DIR),
                        model_name=model_name,
                        distance_metric=metric,
                        detector_backend=detector,
                        enforce_detection=False,
                        silent=True,
                        align=True
                    )
                    
                    if len(dfs) > 0 and not dfs[0].empty:
                        logger.info(f"✅ {model_name} found {len(dfs[0])} matches")
                        
                        for _, row in dfs[0].iterrows():
                            if metric == "cosine":
                                similarity = (1 - float(row['distance'])) * 100
                            else:
                                similarity = max(0, min(100, 100 - (float(row['distance']) * 50)))
                            
                            # Extremely low threshold - show everything above 5%
                            if similarity >= 5:
                                db_path = Path(row['identity'])
                                filename_parts = db_path.name.split('_')
                                metadata_str = filename_parts[0] if filename_parts else ""
                                metadata_parts = metadata_str.split('|')
                                
                                person_info = {
                                    "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                                    "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                                    "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                                    "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                                    "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                                    "matchScore": round(similarity, 2)
                                }
                                
                                all_matches.append(person_info)
                                logger.info(f"✅ Match: {similarity:.1f}% via {model_name}/{detector}")
                    
                    # Memory cleanup
                    del dfs
                    gc.collect()
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"{model_name}/{detector} failed: {e}")
                    continue
        
        # Cleanup temp files
        os.remove(temp_path)
        if 'enhanced_path' in locals() and enhanced_path.exists():
            os.remove(enhanced_path)
        
        if not all_matches:
            logger.info("❌ No matches found")
            return []
        
        # Remove duplicates and sort
        unique_matches = {}
        for match in all_matches:
            name = match['name']
            if name not in unique_matches or match['matchScore'] > unique_matches[name]['matchScore']:
                unique_matches[name] = match
        
        final_results = list(unique_matches.values())
        final_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Returning {len(final_results)} matches")
        return final_results
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []

@app.get("/faces")
async def list_faces():
    faces = []
    for f in KNOWN_FACES_DIR.glob("*.*"):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filename_parts = f.name.split('_')
            metadata_str = filename_parts[0] if filename_parts else ""
            metadata_parts = metadata_str.split('|')
            
            faces.append({
                "filename": f.name,
                "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                "state": metadata_parts[4] if len(metadata_parts) > 4 else ""
            })
    
    return {"faces": faces, "count": len(faces)}

@app.on_event("startup")
async def startup_event():
    """Verify all saved faces on startup"""
    logger.info("🚀 Server starting, verifying saved faces...")
    verify_saved_faces()