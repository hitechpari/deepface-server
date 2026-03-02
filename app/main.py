import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore
import gc
import tempfile
import json
from datetime import datetime

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
# DEEPFACE IMPORT (Lazy Loading)
# ============================================
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported")
except Exception as e:
    logger.error(f"❌ DeepFace import error: {e}")

# ============================================
# CLOUDINARY INIT
# ============================================
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )
    logger.info("✅ Cloudinary configured")
except Exception as e:
    logger.error(f"❌ Cloudinary error: {e}")

# ============================================
# FIREBASE INIT
# ============================================
db = None
try:
    cred_dict = {
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    
    if all([cred_dict["project_id"], cred_dict["private_key"], cred_dict["client_email"]]):
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("✅ Firebase connected")
    else:
        logger.warning("⚠️ Firebase credentials missing")
except Exception as e:
    logger.error(f"❌ Firebase error: {e}")

app = FastAPI(title="Missing Person API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def cleanup():
    gc.collect()

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    cleanup()
    return {
        "message": "Missing Person API",
        "deepface": DEEPFACE_AVAILABLE,
        "firebase": db is not None
    }

@app.get("/health")
async def health():
    cleanup()
    return {"status": "ok"}

@app.post("/add-face-base64")
async def add_face_base64(data: dict):
    """Add face - saves to Cloudinary + Firebase"""
    try:
        image_base64 = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        if not image_base64:
            raise HTTPException(400, "No image")
        
        # Decode image
        image_data = base64.b64decode(image_base64)
        unique_id = str(uuid.uuid4())
        public_id = f"{name}_{age}_{unique_id}".replace(" ", "_")
        
        # Upload to Cloudinary
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        result = cloudinary.uploader.upload(
            tmp_path,
            public_id=public_id,
            folder="missing_persons",
            overwrite=True
        )
        os.unlink(tmp_path)
        
        # Save to Firebase
        if db:
            doc_ref = db.collection('missing_persons').document(unique_id)
            doc_ref.set({
                'name': name,
                'age': age,
                'mobile': mobile,
                'city': city,
                'state': state,
                'image_url': result['secure_url'],
                'timestamp': firestore.SERVER_TIMESTAMP
            })
        
        cleanup()
        logger.info(f"✅ Added: {name}")
        
        return {"success": True, "id": unique_id}
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        cleanup()
        raise HTTPException(500, str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Lightweight search"""
    try:
        image_base64 = data.get('image')
        
        if not image_base64:
            raise HTTPException(400, "No image")
        
        # Save search image
        image_data = base64.b64decode(image_base64)
        search_temp = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        with open(search_temp, "wb") as f:
            f.write(image_data)
        
        if not DEEPFACE_AVAILABLE or not db:
            os.unlink(search_temp)
            return []
        
        # Get all persons
        persons = []
        docs = db.collection('missing_persons').stream()
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            persons.append(data)
        
        if not persons:
            os.unlink(search_temp)
            return []
        
        # Compare with each person (one by one)
        results = []
        import requests
        
        for person in persons:
            try:
                # Download image
                resp = requests.get(person['image_url'], timeout=5)
                if resp.status_code != 200:
                    continue
                
                db_temp = TEMP_DIR / f"{uuid.uuid4()}.jpg"
                with open(db_temp, "wb") as f:
                    f.write(resp.content)
                
                # Compare
                dfs = DeepFace.find(
                    img_path=str(search_temp),
                    db_path=str(db_temp.parent),
                    model_name="Facenet512",
                    enforce_detection=False,
                    silent=True
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    sim = (1 - float(dfs[0].iloc[0]['distance'])) * 100
                    if sim >= 40:
                        results.append({
                            "name": person['name'],
                            "age": person['age'],
                            "mobile": person['mobile'],
                            "city": person['city'],
                            "state": person['state'],
                            "matchScore": round(sim, 2),
                            "photo_url": person['image_url']
                        })
                
                os.unlink(db_temp)
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Error: {e}")
                continue
        
        os.unlink(search_temp)
        cleanup()
        
        # Remove duplicates
        seen = set()
        unique = []
        for r in results:
            if r['name'] not in seen:
                seen.add(r['name'])
                unique.append(r)
        
        unique.sort(key=lambda x: x['matchScore'], reverse=True)
        return unique
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        cleanup()
        return []

@app.get("/faces")
async def list_faces():
    if not db:
        return {"faces": [], "count": 0}
    
    faces = []
    docs = db.collection('missing_persons').stream()
    for doc in docs:
        data = doc.to_dict()
        faces.append({
            "id": doc.id,
            "name": data.get('name'),
            "age": data.get('age'),
            "mobile": data.get('mobile'),
            "city": data.get('city'),
            "state": data.get('state'),
            "photo_url": data.get('image_url')
        })
    
    cleanup()
    return {"faces": faces, "count": len(faces)}