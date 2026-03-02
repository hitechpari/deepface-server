import os
import logging
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import gc

# Set environment variables for memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepFace import with memory optimization
DEEPFACE_AVAILABLE = False
try:
    import tensorflow as tf
    # Limit TensorFlow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        # For CPU, limit thread usage
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    
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

# Global model cache
_MODEL_CACHE = {}

def get_model(model_name):
    """Load model with caching to save memory"""
    global _MODEL_CACHE
    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading model: {model_name}")
        _MODEL_CACHE[model_name] = True  # Just mark as loaded
    return _MODEL_CACHE.get(model_name)

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

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
        
        # Clean up memory
        cleanup_memory()
        
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
    """Search for a face - MEMORY OPTIMIZED VERSION"""
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
        
        # ===== MEMORY OPTIMIZED: Only 2 models instead of 6 =====
        all_matches = []
        
        # Only use 2 most efficient models
        models_to_try = [
            ("Facenet512", "cosine"),      # Most accurate
            ("Facenet512", "euclidean_l2")  # Different metric
        ]
        
        # Single detector to save memory
        detector = "opencv"
        
        for model_name, metric in models_to_try:
            try:
                logger.info(f"🔄 Trying {model_name} with {metric}")
                
                # Force garbage collection before each model
                cleanup_memory()
                
                dfs = DeepFace.find(
                    img_path=str(temp_path),
                    db_path=str(KNOWN_FACES_DIR),
                    model_name=model_name,
                    distance_metric=metric,
                    detector_backend=detector,
                    enforce_detection=False,
                    silent=True,
                    align=True
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    logger.info(f"✅ {model_name} found matches")
                    
                    for _, row in dfs[0].iterrows():
                        # Calculate similarity
                        if metric == "cosine":
                            similarity = (1 - float(row['distance'])) * 100
                        else:
                            similarity = max(0, min(100, 100 - (float(row['distance']) * 50)))
                        
                        # Extract metadata
                        db_path = Path(row['identity'])
                        filename_parts = db_path.name.split('_')
                        metadata_str = filename_parts[0] if filename_parts else ""
                        metadata_parts = metadata_str.split('|')
                        
                        person_info = {
                            "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                            "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                            "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                            "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                            "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                            "matchScore": round(similarity, 2)
                        }
                        
                        all_matches.append(person_info)
                        logger.info(f"✅ Match: {similarity:.1f}%")
                
                # Clean up after each model
                del dfs
                cleanup_memory()
                
            except Exception as e:
                logger.warning(f"{model_name} failed: {e}")
                continue
        
        os.remove(temp_path)
        
        if not all_matches:
            logger.info("❌ No matches found")
            return []
        
        # Remove duplicates
        unique_matches = {}
        for match in all_matches:
            name = match['name']
            if name not in unique_matches or match['matchScore'] > unique_matches[name]['matchScore']:
                unique_matches[name] = match
        
        final_results = list(unique_matches.values())
        final_results.sort(key=lambda x: x['matchScore'], reverse=True)
        
        logger.info(f"✅ Returning {len(final_results)} matches")
        
        # Final cleanup
        cleanup_memory()
        
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