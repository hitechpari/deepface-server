import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np

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
        
        logger.info(f"📸 Add face request: {name}")
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        
        # Create metadata string
        metadata = f"{name}|{age}|{mobile}|{city}|{state}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        file_path = KNOWN_FACES_DIR / filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        logger.info(f"✅ File saved: {filename}")
        
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
    """Search for a face - WITHOUT threshold parameter"""
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
        
        # ===== FIXED: WITHOUT threshold PARAMETER =====
        all_matches = []
        
        # Try different models with different distance metrics
        # Note: threshold parameter is REMOVED - it will use default values
        
        # Model 1: Facenet512 with cosine
        try:
            logger.info("🔄 Trying Facenet512 with cosine")
            dfs = DeepFace.find(
                img_path=str(temp_path),
                db_path=str(KNOWN_FACES_DIR),
                model_name="Facenet512",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True,
                align=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                logger.info(f"✅ Facenet512 found matches")
                for _, row in dfs[0].iterrows():
                    distance = float(row['distance'])
                    similarity = (1 - distance) * 100
                    
                    # Manual threshold - include if similarity > 30%
                    if similarity >= 30:
                        db_path = Path(row['identity'])
                        filename_parts = db_path.name.split('_')
                        metadata_str = filename_parts[0] if filename_parts else ""
                        metadata_parts = metadata_str.split('|')
                        
                        all_matches.append({
                            "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                            "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                            "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                            "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                            "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                            "matchScore": round(similarity, 2)
                        })
                        logger.info(f"✅ Match: {similarity:.1f}%")
        except Exception as e:
            logger.warning(f"Facenet512 failed: {e}")
        
        # Model 2: ArcFace with cosine
        try:
            logger.info("🔄 Trying ArcFace with cosine")
            dfs = DeepFace.find(
                img_path=str(temp_path),
                db_path=str(KNOWN_FACES_DIR),
                model_name="ArcFace",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True,
                align=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                logger.info(f"✅ ArcFace found matches")
                for _, row in dfs[0].iterrows():
                    distance = float(row['distance'])
                    similarity = (1 - distance) * 100
                    
                    if similarity >= 30:
                        db_path = Path(row['identity'])
                        filename_parts = db_path.name.split('_')
                        metadata_str = filename_parts[0] if filename_parts else ""
                        metadata_parts = metadata_str.split('|')
                        
                        all_matches.append({
                            "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                            "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                            "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                            "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                            "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                            "matchScore": round(similarity, 2)
                        })
                        logger.info(f"✅ Match: {similarity:.1f}%")
        except Exception as e:
            logger.warning(f"ArcFace failed: {e}")
        
        # Model 3: VGGFace with cosine
        try:
            logger.info("🔄 Trying VGGFace with cosine")
            dfs = DeepFace.find(
                img_path=str(temp_path),
                db_path=str(KNOWN_FACES_DIR),
                model_name="VGGFace",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True,
                align=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                logger.info(f"✅ VGGFace found matches")
                for _, row in dfs[0].iterrows():
                    distance = float(row['distance'])
                    similarity = (1 - distance) * 100
                    
                    if similarity >= 30:
                        db_path = Path(row['identity'])
                        filename_parts = db_path.name.split('_')
                        metadata_str = filename_parts[0] if filename_parts else ""
                        metadata_parts = metadata_str.split('|')
                        
                        all_matches.append({
                            "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                            "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                            "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                            "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                            "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                            "matchScore": round(similarity, 2)
                        })
                        logger.info(f"✅ Match: {similarity:.1f}%")
        except Exception as e:
            logger.warning(f"VGGFace failed: {e}")
        
        # Model 4: Facenet512 with euclidean_l2
        try:
            logger.info("🔄 Trying Facenet512 with euclidean_l2")
            dfs = DeepFace.find(
                img_path=str(temp_path),
                db_path=str(KNOWN_FACES_DIR),
                model_name="Facenet512",
                distance_metric="euclidean_l2",
                enforce_detection=False,
                silent=True,
                align=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                logger.info(f"✅ Facenet512 (euclidean) found matches")
                for _, row in dfs[0].iterrows():
                    distance = float(row['distance'])
                    similarity = max(0, min(100, 100 - (distance * 100)))
                    
                    if similarity >= 30:
                        db_path = Path(row['identity'])
                        filename_parts = db_path.name.split('_')
                        metadata_str = filename_parts[0] if filename_parts else ""
                        metadata_parts = metadata_str.split('|')
                        
                        all_matches.append({
                            "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                            "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                            "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                            "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                            "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                            "matchScore": round(similarity, 2)
                        })
                        logger.info(f"✅ Match: {similarity:.1f}%")
        except Exception as e:
            logger.warning(f"Facenet512 euclidean failed: {e}")
        
        os.remove(temp_path)
        
        if not all_matches:
            logger.info("❌ No matches found above 30%")
            return []
        
        # Remove duplicates - keep highest score for each person
        unique_matches = {}
        for match in all_matches:
            name = match['name']
            if name not in unique_matches or match['matchScore'] > unique_matches[name]['matchScore']:
                unique_matches[name] = match
        
        final_results = list(unique_matches.values())
        final_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Returning {len(final_results)} unique matches (30% to 100%)")
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
