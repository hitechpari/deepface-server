import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gc
import time

# ============================================
# MAXIMUM MEMORY OPTIMIZATION
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# SIMPLIFIED DEEPFACE IMPORT
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

def cleanup():
    gc.collect()
    time.sleep(0.3)  # More time for memory cleanup

@app.get("/")
async def root():
    return {
        "message": "Missing Person API",
        "deepface": DEEPFACE_AVAILABLE,
        "faces": len(list(KNOWN_FACES_DIR.glob("*.*")))
    }

@app.get("/health")
async def health():
    cleanup()
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
        cleanup()
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search-base64")
async def search_face(data: dict):
    """EXTREME LIGHTWEIGHT SEARCH"""
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
            return []
        
        faces = list(KNOWN_FACES_DIR.glob("*.*"))
        logger.info(f"Faces in DB: {len(faces)}")
        
        if len(faces) == 0:
            os.remove(temp)
            return []
        
        # ===== SIMPLEST POSSIBLE SEARCH =====
        results = []
        
        try:
            logger.info("🔄 Searching...")
            
            # Absolute minimum options - no extras
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
                    
                    # Simple name extraction
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
        
        # Cleanup
        os.remove(temp)
        cleanup()
        
        # Simple dedup
        seen = {}
        for r in results:
            if r['name'] not in seen or r['matchScore'] > seen[r['name']]['matchScore']:
                seen[r['name']] = r
        
        final = list(seen.values())
        final.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Returning {len(final)} matches")
        return final
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return []

@app.get("/faces")
async def list_faces():
    faces = []
    for f in KNOWN_FACES_DIR.glob("*.*"):
        parts = f.name.split('_')[0].split('|')
        faces.append({
            "filename": f.name,
            "name": parts[0] if len(parts) > 0 else "Unknown"
        })
    return {"faces": faces, "count": len(faces)}