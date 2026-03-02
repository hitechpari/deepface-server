import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gc

# ============================================
# MINIMAL SETUP
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# DEEPFACE IMPORT
# ============================================
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported")
except Exception as e:
    logger.error(f"❌ Import failed: {e}")

app = FastAPI(title="Missing Person API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    gc.collect()
    return {
        "message": "Missing Person API",
        "deepface": DEEPFACE_AVAILABLE,
        "faces": len(list(KNOWN_FACES_DIR.glob("*.*")))
    }

@app.get("/health")
async def health():
    gc.collect()
    return {"status": "ok"}

@app.post("/add-face-base64")
async def add_face(data: dict):
    try:
        img = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        if not img:
            raise HTTPException(400, "No image")
        
        img_data = base64.b64decode(img)
        metadata = f"{name}|{age}|{mobile}|{city}|{state}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        path = KNOWN_FACES_DIR / filename
        
        with open(path, "wb") as f:
            f.write(img_data)
        
        logger.info(f"✅ Saved: {filename}")
        gc.collect()
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search-base64")
async def search_face(data: dict):
    """MINIMAL SEARCH - Maximum compatibility"""
    try:
        img = data.get('image')
        
        if not img:
            raise HTTPException(400, "No image")
        
        # Save temp file
        img_data = base64.b64decode(img)
        temp = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        with open(temp, "wb") as f:
            f.write(img_data)
        
        if not DEEPFACE_AVAILABLE:
            os.remove(temp)
            return {"error": "DeepFace not available"}
        
        faces = list(KNOWN_FACES_DIR.glob("*.*"))
        logger.info(f"Faces: {len(faces)}")
        
        if len(faces) == 0:
            os.remove(temp)
            return []
        
        # ===== SIMPLEST POSSIBLE SEARCH =====
        results = []
        
        try:
            logger.info("🔄 Searching...")
            
            # Only essential parameters
            dfs = DeepFace.find(
                img_path=str(temp),
                db_path=str(KNOWN_FACES_DIR),
                model_name="Facenet512",
                enforce_detection=False
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                for _, row in dfs[0].iterrows():
                    sim = (1 - float(row['distance'])) * 100
                    
                    # Get name from filename
                    fname = Path(row['identity']).name
                    parts = fname.split('_')[0].split('|')
                    
                    results.append({
                        "name": parts[0] if len(parts) > 0 else "Unknown",
                        "age": parts[1] if len(parts) > 1 else "",
                        "mobile": parts[2] if len(parts) > 2 else "",
                        "city": parts[3] if len(parts) > 3 else "",
                        "state": parts[4] if len(parts) > 4 else "",
                        "matchScore": round(sim, 2)
                    })
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": str(e)}
        
        # Cleanup
        os.remove(temp)
        gc.collect()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"error": str(e)}

@app.get("/faces")
async def list_faces():
    gc.collect()
    faces = []
    for f in KNOWN_FACES_DIR.glob("*.*"):
        parts = f.name.split('_')[0].split('|')
        faces.append({
            "filename": f.name,
            "name": parts[0] if len(parts) > 0 else "Unknown"
        })
    return {"faces": faces, "count": len(faces)}