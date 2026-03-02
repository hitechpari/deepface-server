import os
import sys
import logging
import shutil
import uuid
import base64
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    from deepface.commons import functions
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
    """Add face from base64 image"""
    try:
        image_base64 = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        current_date = datetime.now().strftime("%Y%m%d")
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        
        # Create metadata string
        metadata = f"{name}|{age}|{mobile}|{city}|{state}|{current_date}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        
        file_path = KNOWN_FACES_DIR / filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        logger.info(f"✅ Face added: {filename}")
        
        return {
            "success": True,
            "message": "Face added successfully",
            "filename": filename
        }
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search for a face with ULTRA TOLERANT matching"""
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
            return {"matched": False, "error": "DeepFace not available"}
        
        known_faces = list(KNOWN_FACES_DIR.glob("*.*"))
        if len(known_faces) == 0:
            os.remove(temp_path)
            return {"matched": False, "message": "No faces in database"}
        
        # ===== ULTRA TOLERANT CONFIGURATION =====
        # Extremely low thresholds for maximum matches
        models_config = [
            # Model, Metric, Threshold (very low = very tolerant)
            ("Facenet512", "cosine", 0.20),      # Most tolerant (was 0.30)
            ("Facenet512", "euclidean_l2", 0.6),  # Very tolerant (was 0.8)
            ("ArcFace", "cosine", 0.25),          # Tolerant (was 0.40)
            ("VGGFace", "cosine", 0.30),          # Added VGGFace for more matches
        ]
        
        # Multiple detectors for better face detection
        detectors = ["mtcnn", "opencv", "retinaface"]
        
        all_matches = []
        
        for detector in detectors:
            for model_name, distance_metric, threshold in models_config:
                try:
                    logger.info(f"🔄 Trying {detector}/{model_name} (threshold: {threshold})")
                    
                    result = DeepFace.find(
                        img_path=str(temp_path),
                        db_path=str(KNOWN_FACES_DIR),
                        model_name=model_name,
                        distance_metric=distance_metric,
                        detector_backend=detector,
                        enforce_detection=False,
                        silent=True,
                        threshold=threshold,
                        normalization="base",
                        align=True
                    )
                    
                    if len(result) > 0 and not result[0].empty:
                        for idx, match in result[0].iterrows():
                            distance = float(match['distance'])
                            
                            # Convert distance to similarity
                            if distance_metric == "cosine":
                                similarity = (1 - distance) * 100
                            else:
                                similarity = max(0, min(100, 100 - (distance * 100)))
                            
                            # VERY LOW THRESHOLD - 40% se upar sab match karo
                            if similarity >= 40:  # Was 50%, now 40%
                                # Extract metadata
                                db_path = Path(match['identity'])
                                filename_parts = db_path.name.split('_')
                                metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
                                metadata_parts = metadata_str.split('|')
                                
                                person_info = {
                                    "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                                    "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                                    "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                                    "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                                    "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                                    "photo_date": metadata_parts[5] if len(metadata_parts) > 5 else "Unknown"
                                }
                                
                                all_matches.append({
                                    "similarity": similarity,
                                    "person_info": person_info,
                                    "model": model_name,
                                    "detector": detector,
                                    "distance": distance
                                })
                                
                                logger.info(f"✅ Match: {person_info['name']} - {similarity:.1f}%")
                    
                except Exception as e:
                    logger.warning(f"Combination {detector}/{model_name} failed: {e}")
                    continue
        
        os.remove(temp_path)
        
        if not all_matches:
            logger.info("❌ No matches found")
            return []
        
        # Sort by similarity
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Group by person and take highest similarity for each
        person_best_matches = {}
        for match in all_matches:
            name = match['person_info']['name']
            if name not in person_best_matches or match['similarity'] > person_best_matches[name]['similarity']:
                person_best_matches[name] = match
        
        # Convert to list and sort again
        unique_matches = list(person_best_matches.values())
        unique_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return ALL matches above 40%, not just top 5
        results = []
        for match in unique_matches:
            if match['similarity'] >= 40:  # 40% se upar sab
                results.append({
                    "name": match['person_info']['name'],
                    "age": match['person_info']['age'],
                    "mobile": match['person_info']['mobile'],
                    "city": match['person_info']['city'],
                    "state": match['person_info']['state'],
                    "matchScore": round(match['similarity'], 2)
                })
        
        logger.info(f"✅ Returning {len(results)} matches")
        return results
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []

@app.get("/faces")
async def list_faces():
    """List all known faces"""
    faces = []
    for f in KNOWN_FACES_DIR.glob("*.*"):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filename_parts = f.name.split('_')
            metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
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
