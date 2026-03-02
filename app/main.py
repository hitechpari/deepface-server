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
from datetime import datetime
import gc
import tempfile
import json

# ============================================
# ENVIRONMENT SETUP
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# FIREBASE INITIALIZATION
# ============================================
try:
    # Firebase credentials from environment
    firebase_cred = {
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    
    cred = credentials.Certificate(firebase_cred)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("✅ Firebase initialized successfully")
except Exception as e:
    logger.error(f"❌ Firebase init error: {e}")
    db = None

# ============================================
# CLOUDINARY INITIALIZATION
# ============================================
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )
    logger.info("✅ Cloudinary initialized successfully")
except Exception as e:
    logger.error(f"❌ Cloudinary init error: {e}")

# ============================================
# DEEPFACE IMPORT
# ============================================
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported")
except Exception as e:
    logger.error(f"❌ DeepFace import error: {e}")

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

# ============================================
# FIREBASE DATABASE FUNCTIONS
# ============================================
async def save_to_firebase(person_data: dict, image_url: str):
    """Save person data to Firebase Firestore"""
    try:
        doc_ref = db.collection('missing_persons').document(person_data['id'])
        doc_ref.set({
            'name': person_data['name'],
            'age': person_data['age'],
            'mobile': person_data['mobile'],
            'city': person_data['city'],
            'state': person_data['state'],
            'image_url': image_url,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'filename': person_data['filename']
        })
        logger.info(f"✅ Saved to Firebase: {person_data['id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Firebase save error: {e}")
        return False

async def get_all_from_firebase():
    """Get all persons from Firebase"""
    try:
        persons = []
        docs = db.collection('missing_persons').stream()
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            persons.append(data)
        logger.info(f"✅ Retrieved {len(persons)} from Firebase")
        return persons
    except Exception as e:
        logger.error(f"❌ Firebase read error: {e}")
        return []

async def get_person_by_id(person_id: str):
    """Get specific person from Firebase"""
    try:
        doc = db.collection('missing_persons').document(person_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error(f"❌ Firebase get error: {e}")
        return None

# ============================================
# CLOUDINARY FUNCTIONS
# ============================================
async def upload_to_cloudinary(image_data: bytes, public_id: str) -> str:
    """Upload image to Cloudinary and return URL"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        result = cloudinary.uploader.upload(
            tmp_path,
            public_id=public_id,
            folder="missing_persons",
            overwrite=True,
            resource_type="image"
        )
        
        os.unlink(tmp_path)
        logger.info(f"✅ Uploaded to Cloudinary")
        return result['secure_url']
        
    except Exception as e:
        logger.error(f"❌ Cloudinary upload error: {e}")
        raise

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    persons = await get_all_from_firebase()
    return {
        "message": "Missing Person API with Firebase",
        "deepface": DEEPFACE_AVAILABLE,
        "firebase_connected": db is not None,
        "faces_in_db": len(persons)
    }

@app.get("/health")
async def health():
    gc.collect()
    return {
        "status": "ok",
        "firebase": db is not None,
        "deepface": DEEPFACE_AVAILABLE
    }

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
        
        # Create unique ID
        unique_id = str(uuid.uuid4())
        filename = f"{name}|{age}|{mobile}|{city}|{state}_{unique_id}.jpg"
        public_id = f"{name}_{age}_{unique_id}".replace(" ", "_")
        
        # Upload to Cloudinary
        image_url = await upload_to_cloudinary(image_data, public_id)
        
        # Save to Firebase
        person_data = {
            'id': unique_id,
            'name': name,
            'age': age,
            'mobile': mobile,
            'city': city,
            'state': state,
            'filename': filename
        }
        
        await save_to_firebase(person_data, image_url)
        
        logger.info(f"✅ Person added: {name}")
        
        return {
            "success": True,
            "message": "Face added successfully",
            "id": unique_id,
            "cloudinary_url": image_url
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search face - compares with all Firebase entries"""
    try:
        image_base64 = data.get('image')
        
        if not image_base64:
            raise HTTPException(400, "No image")
        
        # Decode search image
        search_image_data = base64.b64decode(image_base64)
        search_temp = TEMP_DIR / f"search_{uuid.uuid4()}.jpg"
        with open(search_temp, "wb") as f:
            f.write(search_image_data)
        
        if not DEEPFACE_AVAILABLE or db is None:
            os.remove(search_temp)
            return {"error": "Services not available"}
        
        # Get all persons from Firebase
        persons = await get_all_from_firebase()
        
        if len(persons) == 0:
            os.remove(search_temp)
            return []
        
        # Download and compare each face
        results = []
        
        for person in persons:
            try:
                # Download image from Cloudinary
                import requests
                img_response = requests.get(person['image_url'])
                if img_response.status_code == 200:
                    db_temp = TEMP_DIR / f"db_{uuid.uuid4()}.jpg"
                    with open(db_temp, "wb") as f:
                        f.write(img_response.content)
                    
                    # Compare faces
                    dfs = DeepFace.find(
                        img_path=str(search_temp),
                        db_path=str(db_temp.parent),
                        model_name="Facenet512",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if len(dfs) > 0 and not dfs[0].empty:
                        for _, row in dfs[0].iterrows():
                            similarity = (1 - float(row['distance'])) * 100
                            if similarity >= 40:
                                results.append({
                                    "name": person['name'],
                                    "age": person['age'],
                                    "mobile": person['mobile'],
                                    "city": person['city'],
                                    "state": person['state'],
                                    "matchScore": round(similarity, 2),
                                    "photo_url": person['image_url']
                                })
                    
                    os.unlink(db_temp)
                    
            except Exception as e:
                logger.warning(f"Error processing {person.get('name')}: {e}")
                continue
        
        # Cleanup
        os.unlink(search_temp)
        gc.collect()
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in results:
            if r['name'] not in seen:
                seen.add(r['name'])
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Found {len(unique_results)} matches")
        return unique_results
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []

@app.get("/faces")
async def list_faces():
    """List all faces from Firebase"""
    persons = await get_all_from_firebase()
    
    faces = []
    for person in persons:
        faces.append({
            "id": person.get('id'),
            "name": person['name'],
            "age": person['age'],
            "mobile": person['mobile'],
            "city": person['city'],
            "state": person['state'],
            "photo_url": person['image_url']
        })
    
    return {"faces": faces, "count": len(faces)}

@app.delete("/face/{person_id}")
async def delete_face(person_id: str):
    """Delete person from Firebase"""
    try:
        db.collection('missing_persons').document(person_id).delete()
        return {"success": True, "message": f"Deleted {person_id}"}
    except Exception as e:
        raise HTTPException(500, str(e))