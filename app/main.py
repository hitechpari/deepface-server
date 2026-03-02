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

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepFace import
DEEPFACE_AVAILABLE = False
try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    from deepface import DeepFace
    from deepface.detectors import FaceDetector
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported successfully")
except Exception as e:
    logger.error(f"❌ DeepFace import error: {e}")

app = FastAPI(title="Missing Person Face Recognition API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
KNOWN_FACES_DIR = Path("app/known_faces")
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def check_face_detection(image_path):
    """Check if face is detected in image"""
    try:
        # Try multiple detectors
        detectors = ["opencv", "mtcnn", "retinaface"]
        for detector in detectors:
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=str(image_path),
                    detector_backend=detector,
                    enforce_detection=False
                )
                if len(face_objs) > 0:
                    logger.info(f"✅ Face detected with {detector}")
                    return True
            except:
                continue
        logger.warning("❌ No face detected in image")
        return False
    except Exception as e:
        logger.warning(f"Face detection check failed: {e}")
        return False

def enhance_image(image_path):
    """Enhance image for better face detection"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Save enhanced image
        enhanced_path = image_path.parent / f"enhanced_{image_path.name}"
        cv2.imwrite(str(enhanced_path), enhanced_bgr)
        
        logger.info(f"✅ Image enhanced: {enhanced_path}")
        return enhanced_path
    except Exception as e:
        logger.warning(f"Image enhancement failed: {e}")
        return None

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
    try:
        image_base64 = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        image_data = base64.b64decode(image_base64)
        metadata = f"{name}|{age}|{mobile}|{city}|{state}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        file_path = KNOWN_FACES_DIR / filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        logger.info(f"✅ File saved: {filename}")
        
        # Check if face is detectable in saved image
        has_face = check_face_detection(file_path)
        if not has_face:
            # Try to enhance image
            enhanced_path = enhance_image(file_path)
            if enhanced_path:
                # Replace with enhanced version
                enhanced_path.replace(file_path)
                logger.info("✅ Replaced with enhanced image")
        
        return {"success": True, "message": "Face added successfully"}
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search for a face - WITH FACE DETECTION VERIFICATION"""
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
        
        # Check if query image has detectable face
        logger.info("🔍 Checking query image for face...")
        has_face = check_face_detection(temp_path)
        
        if not has_face:
            logger.warning("⚠️ No face detected in query image, trying enhancement...")
            enhanced_path = enhance_image(temp_path)
            if enhanced_path:
                search_path = enhanced_path
                logger.info("✅ Using enhanced image for search")
            else:
                search_path = temp_path
        else:
            search_path = temp_path
        
        known_faces = list(KNOWN_FACES_DIR.glob("*.*"))
        logger.info(f"Known faces count: {len(known_faces)}")
        
        if len(known_faces) == 0:
            os.remove(temp_path)
            if enhanced_path and enhanced_path.exists():
                os.remove(enhanced_path)
            return []
        
        # ===== TRY MULTIPLE DETECTORS =====
        all_matches = []
        
        # Try different detectors
        detectors = ["opencv", "mtcnn", "retinaface"]
        
        for detector in detectors:
            try:
                logger.info(f"🔄 Using detector: {detector}")
                
                # Try with Facenet512
                dfs = DeepFace.find(
                    img_path=str(search_path),
                    db_path=str(KNOWN_FACES_DIR),
                    model_name="Facenet512",
                    distance_metric="cosine",
                    detector_backend=detector,
                    enforce_detection=False,
                    silent=True,
                    align=True
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    logger.info(f"✅ Facenet512 found matches with {detector}")
                    
                    for _, row in dfs[0].iterrows():
                        similarity = (1 - float(row['distance'])) * 100
                        
                        if similarity >= 10:  # Extremely low threshold
                            # Extract metadata
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
                            logger.info(f"✅ Match: {similarity:.1f}%")
                            
            except Exception as e:
                logger.warning(f"Detector {detector} failed: {e}")
                continue
        
        # Clean up
        os.remove(temp_path)
        if 'enhanced_path' in locals() and enhanced_path and enhanced_path.exists():
            os.remove(enhanced_path)
        
        if not all_matches:
            logger.info("❌ No matches found")
            return []
        
        # Remove duplicates
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