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
import requests

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
# CLOUDINARY INITIALIZATION
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
# FIREBASE INITIALIZATION
# ============================================
db = None
try:
    # Get credentials from environment variables
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    private_key = os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n')
    client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
    
    if project_id and private_key and client_email:
        cred_dict = {
            "type": "service_account",
            "project_id": project_id,
            "private_key": private_key,
            "client_email": client_email,
            "token_uri": "https://oauth2.googleapis.com/token"
        }
        
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("✅ Firebase connected successfully")
        
        # Test connection by trying to list collections
        collections = list(db.collections())
        logger.info(f"📁 Firebase collections: {len(collections)}")
    else:
        logger.warning("⚠️ Firebase credentials missing")
        
except Exception as e:
    logger.error(f"❌ Firebase initialization error: {e}")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(title="Missing Person Face Recognition API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DIRECTORY SETUP
# ============================================
TEMP_DIR = Path("/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================
def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
async def root():
    cleanup_memory()
    return {
        "message": "Missing Person Face Recognition API",
        "status": "healthy",
        "deepface_available": DEEPFACE_AVAILABLE,
        "firebase_connected": db is not None,
        "timestamp": str(uuid.uuid4())
    }

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.get("/health")
async def health_check():
    cleanup_memory()
    return {
        "status": "healthy",
        "deepface_available": DEEPFACE_AVAILABLE,
        "firebase_connected": db is not None
    }

# ============================================
# ADD FACE ENDPOINT
# ============================================
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
        
        # Validation
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        if not name or not age or not mobile or not city or not state:
            raise HTTPException(status_code=400, detail="All fields are required")
        
        logger.info(f"📸 Adding face for: {name}")
        
        # Decode image
        image_data = base64.b64decode(image_base64)
        unique_id = str(uuid.uuid4())
        public_id = f"{name}_{age}_{unique_id}".replace(" ", "_").replace("|", "_")
        
        # Upload to Cloudinary
        logger.info("☁️ Uploading to Cloudinary...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        try:
            result = cloudinary.uploader.upload(
                tmp_path,
                public_id=public_id,
                folder="missing_persons",
                overwrite=True,
                resource_type="image"
            )
            image_url = result['secure_url']
            logger.info(f"✅ Cloudinary upload successful: {image_url[:50]}...")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Save to Firebase
        if db:
            logger.info("🔥 Saving to Firebase...")
            doc_ref = db.collection('missing_persons').document(unique_id)
            doc_data = {
                'name': name,
                'age': age,
                'mobile': mobile,
                'city': city,
                'state': state,
                'image_url': image_url,
                'timestamp': firestore.SERVER_TIMESTAMP
            }
            doc_ref.set(doc_data)
            logger.info("✅ Firebase save successful")
        else:
            logger.warning("⚠️ Firebase not available, skipping database save")
        
        cleanup_memory()
        logger.info(f"✅ Person added successfully: {name}")
        
        return {
            "success": True,
            "message": "Face added successfully",
            "id": unique_id,
            "image_url": image_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding face: {e}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# SEARCH FACE ENDPOINT
# ============================================
@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search for a face in the database"""
    try:
        image_base64 = data.get('image')
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        logger.info("🔍 Search request received")
        
        # Decode and save search image
        image_data = base64.b64decode(image_base64)
        search_temp = TEMP_DIR / f"search_{uuid.uuid4()}.jpg"
        with open(search_temp, "wb") as f:
            f.write(image_data)
        
        # Check if DeepFace is available
        if not DEEPFACE_AVAILABLE:
            os.unlink(search_temp)
            return {"error": "DeepFace not available"}
        
        # Check if Firebase is available
        if not db:
            os.unlink(search_temp)
            return {"error": "Firebase not available"}
        
        # Get all persons from Firebase
        logger.info("📊 Fetching persons from Firebase...")
        persons = []
        docs = db.collection('missing_persons').stream()
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            persons.append(data)
        
        logger.info(f"📊 Found {len(persons)} persons in database")
        
        if not persons:
            os.unlink(search_temp)
            return []
        
        # Compare with each person
        results = []
        
        for idx, person in enumerate(persons):
            try:
                logger.info(f"🔄 Comparing with person {idx+1}/{len(persons)}: {person.get('name', 'Unknown')}")
                
                # Check if image_url exists
                if 'image_url' not in person:
                    logger.warning(f"⚠️ Person {person.get('name', 'Unknown')} has no image_url")
                    continue
                
                # Download person's image from Cloudinary
                resp = requests.get(person['image_url'], timeout=10)
                if resp.status_code != 200:
                    logger.warning(f"⚠️ Failed to download image for {person.get('name', 'Unknown')}")
                    continue
                
                # Save to temp file
                db_temp = TEMP_DIR / f"db_{uuid.uuid4()}.jpg"
                with open(db_temp, "wb") as f:
                    f.write(resp.content)
                
                # Compare faces
                try:
                    dfs = DeepFace.find(
                        img_path=str(search_temp),
                        db_path=str(db_temp.parent),
                        model_name="Facenet512",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if len(dfs) > 0 and not dfs[0].empty:
                        # Get the match for this specific image
                        for _, row in dfs[0].iterrows():
                            # Check if this match corresponds to our current person
                            if str(db_temp) in str(row['identity']):
                                similarity = (1 - float(row['distance'])) * 100
                                if similarity >= 40:  # 40% threshold
                                    results.append({
                                        "name": person.get('name', 'Unknown'),
                                        "age": person.get('age', ''),
                                        "mobile": person.get('mobile', ''),
                                        "city": person.get('city', ''),
                                        "state": person.get('state', ''),
                                        "matchScore": round(similarity, 2),
                                        "photo_url": person.get('image_url', '')
                                    })
                                    logger.info(f"✅ Match found: {person.get('name')} - {similarity:.1f}%")
                                break
                except Exception as e:
                    logger.warning(f"⚠️ Face comparison error for {person.get('name')}: {e}")
                
                # Clean up temp file
                if os.path.exists(db_temp):
                    os.unlink(db_temp)
                
                # Periodic cleanup
                if idx % 5 == 0:
                    cleanup_memory()
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing {person.get('name', 'Unknown')}: {e}")
                continue
        
        # Clean up search temp file
        if os.path.exists(search_temp):
            os.unlink(search_temp)
        
        # Remove duplicates and sort
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r['name']}_{r['mobile']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Found {len(unique_results)} unique matches")
        cleanup_memory()
        
        return unique_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# LIST FACES ENDPOINT
# ============================================
@app.get("/faces")
async def list_faces():
    """List all known faces from Firebase"""
    try:
        if not db:
            return {"faces": [], "count": 0, "error": "Firebase not available"}
        
        logger.info("📋 Listing all faces from Firebase...")
        
        faces = []
        docs = db.collection('missing_persons').stream()
        for doc in docs:
            data = doc.to_dict()
            faces.append({
                "id": doc.id,
                "name": data.get('name', 'Unknown'),
                "age": data.get('age', ''),
                "mobile": data.get('mobile', ''),
                "city": data.get('city', ''),
                "state": data.get('state', ''),
                "photo_url": data.get('image_url', ''),
                "timestamp": str(data.get('timestamp', ''))
            })
        
        logger.info(f"✅ Found {len(faces)} faces")
        cleanup_memory()
        
        return {"faces": faces, "count": len(faces)}
        
    except Exception as e:
        logger.error(f"❌ Error listing faces: {e}")
        cleanup_memory()
        return {"faces": [], "count": 0, "error": str(e)}

# ============================================
# DELETE FACE ENDPOINT
# ============================================
@app.delete("/face/{person_id}")
async def delete_face(person_id: str):
    """Delete a person from Firebase"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Firebase not available")
        
        logger.info(f"🗑️ Deleting person: {person_id}")
        
        # Get the document first to get image_url
        doc_ref = db.collection('missing_persons').document(person_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Delete from Firebase
        doc_ref.delete()
        
        logger.info(f"✅ Deleted person: {person_id}")
        cleanup_memory()
        
        return {"success": True, "message": "Person deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting face: {e}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# STARTUP EVENT
# ============================================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Server starting...")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")
    logger.info(f"🔥 Firebase connected: {db is not None}")
    logger.info(f"🤖 DeepFace available: {DEEPFACE_AVAILABLE}")