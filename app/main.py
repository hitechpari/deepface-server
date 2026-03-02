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
    """Search for a face with improved matching"""
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
        
        # ===== IMPROVED MATCHING CONFIGURATION =====
        # Lower thresholds for better matching
        models_config = [
            # Model, Metric, Threshold (lower = more matches)
            ("Facenet512", "cosine", 0.30),      # Most tolerant
            ("Facenet512", "euclidean_l2", 0.8),  # L2 normalized
            ("ArcFace", "cosine", 0.40),          # Good for variations
        ]
        
        detector_backend = "mtcnn"
        best_matches = []
        
        for model_name, distance_metric, threshold in models_config:
            try:
                logger.info(f"🔄 Trying {model_name} with threshold {threshold}")
                
                result = DeepFace.find(
                    img_path=str(temp_path),
                    db_path=str(KNOWN_FACES_DIR),
                    model_name=model_name,
                    distance_metric=distance_metric,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    silent=True,
                    threshold=threshold,  # Using lower threshold
                    normalization="base",
                    align=True
                )
                
                if len(result) > 0 and not result[0].empty:
                    for idx, match in result[0].iterrows():
                        distance = float(match['distance'])
                        
                        # Convert distance to similarity percentage
                        if distance_metric == "cosine":
                            similarity = (1 - distance) * 100
                        else:
                            similarity = max(0, min(100, 100 - (distance * 100)))
                        
                        # Only include if similarity > 50%
                        if similarity >= 50:
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
                            
                            best_matches.append({
                                "similarity": similarity,
                                "person_info": person_info,
                                "model": model_name,
                                "distance": distance
                            })
                            
                            logger.info(f"✅ Match: {person_info['name']} - {similarity:.1f}%")
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        os.remove(temp_path)
        
        # Sort by similarity (highest first)
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Remove duplicates (same person from different models)
        unique_matches = []
        seen_names = set()
        for match in best_matches:
            name = match['person_info']['name']
            if name not in seen_names:
                seen_names.add(name)
                unique_matches.append(match)
        
        if unique_matches:
            # Return top matches
            results = []
            for match in unique_matches[:5]:  # Top 5 matches
                results.append({
                    "name": match['person_info']['name'],
                    "age": match['person_info']['age'],
                    "mobile": match['person_info']['mobile'],
                    "city": match['person_info']['city'],
                    "state": match['person_info']['state'],
                    "matchScore": round(match['similarity'], 2)
                })
            
            return results
        else:
            return []
    
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
