import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
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
    from deepface import DeepFace
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
    """Search for a face using 4 different models - shows 40% to 100% matches"""
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
        
        known_faces = list(KNOWN_FACES_DIR.glob("*.*"))
        if len(known_faces) == 0:
            os.remove(temp_path)
            return []
        
        # ===== 4 DIFFERENT MODELS WITH DIFFERENT THRESHOLDS =====
        all_matches = []
        
        # Model configurations: (model_name, distance_metric, threshold)
        models_config = [
            # Model 1: Facenet512 - Most accurate, low threshold for 40% matches
            ("Facenet512", "cosine", 0.20),
            
            # Model 2: ArcFace - Best for age-invariant, medium threshold
            ("ArcFace", "cosine", 0.25),
            
            # Model 3: VGGFace - Good all-rounder
            ("VGGFace", "cosine", 0.28),
            
            # Model 4: Facenet with Euclidean - Different metric for variety
            ("Facenet512", "euclidean_l2", 0.7),
        ]
        
        # Try each model configuration
        for model_name, metric, threshold in models_config:
            try:
                logger.info(f"🔄 Trying {model_name} with {metric}, threshold={threshold}")
                
                dfs = DeepFace.find(
                    img_path=str(temp_path),
                    db_path=str(KNOWN_FACES_DIR),
                    model_name=model_name,
                    distance_metric=metric,
                    enforce_detection=False,
                    silent=True,
                    threshold=threshold,
                    align=True
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    for _, row in dfs[0].iterrows():
                        # Calculate similarity percentage
                        if metric == "cosine":
                            similarity = (1 - float(row['distance'])) * 100
                        else:  # euclidean_l2
                            similarity = max(0, min(100, 100 - (float(row['distance']) * 100)))
                        
                        # Include ALL matches from 40% to 100%
                        if similarity >= 40:
                            # Extract metadata from filename
                            db_path = Path(row['identity'])
                            filename_parts = db_path.name.split('_')
                            metadata_str = filename_parts[0] if filename_parts else ""
                            metadata_parts = metadata_str.split('|')
                            
                            match_data = {
                                "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                                "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                                "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                                "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                                "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                                "matchScore": round(similarity, 2),
                                "model": model_name  # Track which model found it
                            }
                            all_matches.append(match_data)
                            
                            logger.info(f"✅ Match: {match_data['name']} - {similarity:.1f}% (via {model_name})")
                            
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        os.remove(temp_path)
        
        if not all_matches:
            logger.info("❌ No matches found above 40%")
            return []
        
        # Remove duplicates - keep highest score for each person
        unique_matches = {}
        for match in all_matches:
            name = match['name']
            if name not in unique_matches or match['matchScore'] > unique_matches[name]['matchScore']:
                unique_matches[name] = match
        
        # Convert to list and sort by score (highest first)
        final_results = list(unique_matches.values())
        final_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        # Remove model field before sending to client
        for result in final_results:
            if 'model' in result:
                del result['model']
        
        logger.info(f"✅ Returning {len(final_results)} unique matches (40% to 100%)")
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