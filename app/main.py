from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import uuid
import base64
from PIL import Image
import io
import numpy as np

# DeepFace import with error handling
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError as e:
    print(f"DeepFace import error: {e}")
    DEEPFACE_AVAILABLE = False

app = FastAPI(title="Missing Person Face Recognition API")

# CORS setup - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Known faces directory
KNOWN_FACES_DIR = Path("app/known_faces")
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

# Temporary directory for uploads
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
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        return {
            "success": True,
            "message": "Face added successfully",
            "filename": filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search for a face from base64 image"""
    try:
        image_base64 = data.get('image')
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        
        # Save temporary file
        temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as buffer:
            buffer.write(image_data)
        
        if not DEEPFACE_AVAILABLE:
            os.remove(temp_path)
            return {
                "matched": False,
                "error": "DeepFace not available",
                "fallback": True
            }
        
        known_faces = list(KNOWN_FACES_DIR.glob("*.*"))
        
        if len(known_faces) == 0:
            os.remove(temp_path)
            return {"matched": False, "message": "No faces in database"}
        
        try:
            from deepface import DeepFace
            result = DeepFace.find(
                img_path=str(temp_path),
                db_path=str(KNOWN_FACES_DIR),
                model_name="Facenet512",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True
            )
        except Exception as deepface_error:
            os.remove(temp_path)
            return {
                "matched": False,
                "error": str(deepface_error),
                "fallback": True
            }
        
        os.remove(temp_path)
        
        if len(result) > 0 and not result[0].empty:
            match = result[0].iloc[0]
            identity_path = Path(match['identity'])
            
            filename_parts = identity_path.name.split('_')
            metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
            metadata_parts = metadata_str.split('|')
            
            person_info = {
                "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                "state": metadata_parts[4] if len(metadata_parts) > 4 else ""
            }
            
            similarity = (1 - float(match['distance'])) * 100
            
            return {
                "matched": True,
                "identity": identity_path.name,
                "person_info": person_info,
                "similarity": f"{similarity:.2f}%",
                "match_score": round(similarity, 2)
            }
        else:
            return {"matched": False, "message": "No match found"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
