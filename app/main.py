import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gc

# ============================================
# MINIMAL SETUP - NO EXTRA IMPORTS
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# SIMPLE DEEPFACE IMPORT
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

# ============================================
# SIMPLE API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "Missing Person API",
        "deepface": DEEPFACE_AVAILABLE,
        "faces": len(list(KNOWN_FACES_DIR.glob("*.*")))
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": str(uuid.uuid4())}

@app.post("/add-face-base64")
async def add_face(data: dict):
    """Add face - simple version"""
    try:
        img = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        if not img:
            raise HTTPException(400, "No image")
        
        # Save image
        img_data = base64.b64decode(img)
        metadata = f"{name}|{age}|{mobile}|{city}|{state}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        file_path = KNOWN_FACES_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(img_data)
        
        logger.info(f"✅ Saved: {filename}")
        gc.collect()
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search-base64")
async def search_face(data: dict):
    """Search face - ultra simple"""
    try:
        img = data.get('image')
        
        if not img:
            raise HTTPException(400, "No image")
        
        # Save temp file
        img_data = base64.b64decode(img)
        temp = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        with open(temp, "wb") as f:
            f.write(img_data)
        
        # Check faces
        faces = list(KNOWN_FACES_DIR.glob("*.*"))
        logger.info(f"Faces: {len(faces)}")
        
        if len(faces) == 0:
            os.remove(temp)
            return []
        
        # ===== SINGLE MODEL SEARCH =====
        results = []
        
        try:
            logger.info("🔄 Searching...")
            
            dfs = DeepFace.find(
                img_path=str(temp),
                db_path=str(KNOWN_FACES_DIR),
                model_name="Facenet512",
                enforce_detection=False,
                silent=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                logger.info(f"✅ Found matches")
                
                for _, row in dfs[0].iterrows():
                    sim = (1 - float(row['distance'])) * 100
                    
                    # Extract name from filename
                    db_path = Path(row['identity'])
                    name_parts = db_path.name.split('_')[0].split('|')
                    
                    results.append({
                        "name": name_parts[0] if len(name_parts) > 0 else "Unknown",
                        "age": name_parts[1] if len(name_parts) > 1 else "",
                        "mobile": name_parts[2] if len(name_parts) > 2 else "",
                        "city": name_parts[3] if len(name_parts) > 3 else "",
                        "state": name_parts[4] if len(name_parts) > 4 else "",
                        "matchScore": round(sim, 2)
                    })
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
        
        # Cleanup
        os.remove(temp)
        gc.collect()
        
        # Remove duplicates
        seen = set()
        unique = []
        for r in results:
            if r['name'] not in seen:
                seen.add(r['name'])
                unique.append(r)
        
        unique.sort(key=lambda x: x['matchScore'], reverse=True)
        logger.info(f"✅ Returning {len(unique)} matches")
        return unique
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return []

@app.get("/faces")
async def list_faces():
    """List all faces"""
    faces = []
    for f in KNOWN_FACES_DIR.glob("*.*"):
        name_parts = f.name.split('_')[0].split('|')
        faces.append({
            "filename": f.name,
            "name": name_parts[0] if len(name_parts) > 0 else "Unknown"
        })
    return {"faces": faces, "count": len(faces)}

@app.on_event("startup")
async def startup():
    """Startup log"""
    logger.info("🚀 Server started")
    logger.info(f"📁 Faces dir: {KNOWN_FACES_DIR}")
    logger.info(f"📁 Faces count: {len(list(KNOWN_FACES_DIR.glob('*.*')))}")