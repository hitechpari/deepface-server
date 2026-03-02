import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import io

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepFace import
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    from deepface.commons import distance as dst
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

def preprocess_image(image_path):
    """Preprocess image to improve face detection"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Enhance image
        # 1. Denoise
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
        
        # 2. Increase contrast
        lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l,a,b])
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Save enhanced image
        enhanced_path = image_path.parent / f"enhanced_{image_path.name}"
        cv2.imwrite(str(enhanced_path), cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))
        
        return enhanced_path
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
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
    """Add face from base64 image"""
    try:
        image_base64 = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        logger.info(f"📸 Add face request: {name}")
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        logger.info(f"Image decoded, size: {len(image_data)} bytes")
        
        # Create metadata string
        metadata = f"{name}|{age}|{mobile}|{city}|{state}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        file_path = KNOWN_FACES_DIR / filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        logger.info(f"✅ File saved: {filename}")
        
        # Try to detect face in saved image
        try:
            face_objs = DeepFace.extract_faces(
                img_path=str(file_path),
                detector_backend="opencv",
                enforce_detection=False
            )
            logger.info(f"Face detection in saved image: {len(face_objs)} faces found")
        except Exception as e:
            logger.warning(f"Face detection in saved image failed: {e}")
        
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
    """Search for a face - ULTIMATE TOLERANT VERSION"""
    try:
        image_base64 = data.get('image')
        
        logger.info("🔍 Search request received")
        
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
        
        known_faces = list(KNOWN_FACES_DIR.glob("*.*"))
        logger.info(f"Known faces count: {len(known_faces)}")
        
        if len(known_faces) == 0:
            os.remove(temp_path)
            return []
        
        # ===== ULTIMATE TOLERANT CONFIGURATION =====
        all_matches = []
        
        # Try preprocessing if face detection fails
        enhanced_path = preprocess_image(temp_path)
        if enhanced_path and enhanced_path.exists():
            logger.info("✅ Image preprocessed for better detection")
            search_paths = [temp_path, enhanced_path]
        else:
            search_paths = [temp_path]
        
        # Multiple detectors
        detectors = [
            "opencv",      # Fast, works for most cases
            "mtcnn",       # Good for small faces
            "retinaface",  # Best accuracy
            "dlib",        # Reliable
            "ssd"          # Alternative
        ]
        
        # Multiple models
        models = [
            ("Facenet512", "cosine"),
            ("Facenet512", "euclidean_l2"),
            ("ArcFace", "cosine"),
            ("VGGFace", "cosine"),
            ("Dlib", "cosine"),
            ("OpenFace", "cosine"),
        ]
        
        total_attempts = 0
        for search_path in search_paths:
            for detector in detectors:
                for model_name, metric in models:
                    try:
                        total_attempts += 1
                        logger.info(f"🔄 Attempt {total_attempts}: {detector}/{model_name}/{metric}")
                        
                        dfs = DeepFace.find(
                            img_path=str(search_path),
                            db_path=str(KNOWN_FACES_DIR),
                            model_name=model_name,
                            distance_metric=metric,
                            detector_backend=detector,
                            enforce_detection=False,  # Don't fail if no face detected
                            silent=True,
                            align=True,
                            normalization="base"
                        )
                        
                        if len(dfs) > 0 and not dfs[0].empty:
                            logger.info(f"✅ Match found with {detector}/{model_name}")
                            
                            for _, row in dfs[0].iterrows():
                                # Calculate similarity
                                if metric == "cosine":
                                    similarity = (1 - float(row['distance'])) * 100
                                else:
                                    similarity = max(0, min(100, 100 - (float(row['distance']) * 50)))
                                
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
                                logger.info(f"✅ Match: {person_info['name']} - {similarity:.1f}%")
                                
                    except Exception as e:
                        logger.debug(f"{detector}/{model_name} failed: {e}")
                        continue
        
        # Clean up
        os.remove(temp_path)
        if enhanced_path and enhanced_path.exists():
            os.remove(enhanced_path)
        
        if not all_matches:
            logger.info("❌ No matches found after 36 attempts")
            return []
        
        # Remove duplicates
        unique_matches = {}
        for match in all_matches:
            name = match['name']
            if name not in unique_matches or match['matchScore'] > unique_matches[name]['matchScore']:
                unique_matches[name] = match
        
        final_results = list(unique_matches.values())
        final_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Returning {len(final_results)} matches after {total_attempts} attempts")
        return final_results
    
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