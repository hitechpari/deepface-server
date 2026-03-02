import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
from datetime import datetime
import gc
import tempfile

# ============================================
# ENVIRONMENT SETUP
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepFace import
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

# Temporary directory for processing
TEMP_DIR = Path("/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# DATABASE: Cloudinary URLs store karenge
# ============================================
# Ab hum filesystem use nahi karenge, sirf Cloudinary
# Photos Cloudinary par save hongi aur unke URLs
# is dictionary mein store honge (in-memory)
# NOTE: Real app mein yeh database (PostgreSQL/MongoDB) use karo
face_database = {}  # { "filename": { "url": "...", "metadata": {...} } }

# ============================================
# CLOUDINARY FUNCTIONS
# ============================================
async def upload_to_cloudinary(image_data: bytes, public_id: str) -> str:
    """Upload image to Cloudinary and return URL"""
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            tmp_path,
            public_id=public_id,
            folder="missing_persons",
            overwrite=True,
            resource_type="image"
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        logger.info(f"✅ Uploaded to Cloudinary: {result['secure_url']}")
        return result['secure_url']
        
    except Exception as e:
        logger.error(f"❌ Cloudinary upload error: {e}")
        raise

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    return {
        "message": "Missing Person API with Cloudinary",
        "deepface": DEEPFACE_AVAILABLE,
        "faces_in_db": len(face_database)
    }

@app.get("/health")
async def health():
    gc.collect()
    return {"status": "ok"}

@app.post("/add-face-base64")
async def add_face_base64(data: dict):
    """Add face - saves to Cloudinary"""
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
        public_id = f"{name}_{age}_{unique_id}".replace(" ", "_")
        
        # Upload to Cloudinary
        image_url = await upload_to_cloudinary(image_data, public_id)
        
        # Store metadata with URL
        filename = f"{name}|{age}|{mobile}|{city}|{state}_{unique_id}.jpg"
        face_database[filename] = {
            "url": image_url,
            "name": name,
            "age": age,
            "mobile": mobile,
            "city": city,
            "state": state,
            "filename": filename
        }
        
        logger.info(f"✅ Face added: {filename}")
        logger.info(f"📸 Cloudinary URL: {image_url}")
        
        return {
            "success": True,
            "message": "Face added successfully",
            "filename": filename,
            "cloudinary_url": image_url
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search face - downloads from Cloudinary for comparison"""
    try:
        image_base64 = data.get('image')
        
        if not image_base64:
            raise HTTPException(400, "No image")
        
        # Decode search image
        search_image_data = base64.b64decode(image_base64)
        search_temp = TEMP_DIR / f"search_{uuid.uuid4()}.jpg"
        with open(search_temp, "wb") as f:
            f.write(search_image_data)
        
        if not DEEPFACE_AVAILABLE:
            os.remove(search_temp)
            return {"error": "DeepFace not available"}
        
        if len(face_database) == 0:
            os.remove(search_temp)
            return []
        
        # Download all faces from Cloudinary to temp for comparison
        results = []
        
        for filename, data in face_database.items():
            try:
                # Download image from Cloudinary
                import requests
                img_response = requests.get(data['url'])
                if img_response.status_code == 200:
                    db_temp = TEMP_DIR / f"db_{uuid.uuid4()}.jpg"
                    with open(db_temp, "wb") as f:
                        f.write(img_response.content)
                    
                    # Compare faces
                    dfs = DeepFace.find(
                        img_path=str(search_temp),
                        db_path=str(db_temp.parent),  # Directory containing temp file
                        model_name="Facenet512",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if len(dfs) > 0 and not dfs[0].empty:
                        for _, row in dfs[0].iterrows():
                            similarity = (1 - float(row['distance'])) * 100
                            if similarity >= 40:
                                results.append({
                                    "name": data['name'],
                                    "age": data['age'],
                                    "mobile": data['mobile'],
                                    "city": data['city'],
                                    "state": data['state'],
                                    "matchScore": round(similarity, 2),
                                    "photo_url": data['url']
                                })
                    
                    # Clean up temp db file
                    os.unlink(db_temp)
                    
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
                continue
        
        # Clean up search temp file
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
    """List all faces from Cloudinary"""
    faces = []
    for filename, data in face_database.items():
        faces.append({
            "filename": filename,
            "name": data['name'],
            "age": data['age'],
            "mobile": data['mobile'],
            "city": data['city'],
            "state": data['state'],
            "cloudinary_url": data['url']
        })
    
    return {"faces": faces, "count": len(faces)}