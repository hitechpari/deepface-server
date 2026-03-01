import os
import sys
import logging
import shutil
import uuid
import base64
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from datetime import datetime

# Set environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import DeepFace with proper error handling
DEEPFACE_AVAILABLE = False
try:
    import tensorflow as tf
    logger.info(f"✅ TensorFlow version: {tf.__version__}")
    
    # Import deepface
    from deepface import DeepFace
    from deepface.commons import functions
    from deepface.detectors import FaceDetector
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported successfully")
    
except ImportError as e:
    logger.error(f"❌ DeepFace import error: {e}")
except Exception as e:
    logger.error(f"❌ Other error: {e}")

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

# Check directory permissions
logger.info(f"📁 Known faces directory: {KNOWN_FACES_DIR.absolute()}")
logger.info(f"📁 Directory exists: {KNOWN_FACES_DIR.exists()}")
logger.info(f"📁 Directory writable: {os.access(KNOWN_FACES_DIR, os.W_OK)}")

@app.get("/")
async def root():
    """Root endpoint - server info"""
    return {
        "message": "Missing Person Face Recognition API",
        "status": "healthy",
        "deepface_available": DEEPFACE_AVAILABLE,
        "known_faces_count": len(list(KNOWN_FACES_DIR.glob("*.*")))
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "deepface_available": DEEPFACE_AVAILABLE,
        "known_faces_count": len(list(KNOWN_FACES_DIR.glob("*.*"))),
        "timestamp": str(uuid.uuid4())
    }

@app.post("/add-face-base64")
async def add_face_base64(data: dict):
    """Add face from base64 image with enhanced metadata for age tracking"""
    try:
        image_base64 = data.get('image')
        name = data.get('name')
        age = data.get('age')
        mobile = data.get('mobile')
        city = data.get('city')
        state = data.get('state')
        
        # Add timestamp for age tracking
        current_date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"✅ Add face request received for: {name}")
        logger.info(f"Image base64 length: {len(image_base64) if image_base64 else 0}")
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        logger.info(f"Image decoded, size: {len(image_data)} bytes")
        
        # Create metadata string with age and date for tracking
        # Format: Name|Age|Mobile|City|State|Date
        metadata = f"{name}|{age}|{mobile}|{city}|{state}|{current_date}"
        filename = f"{metadata}_{uuid.uuid4()}.jpg"
        
        file_path = KNOWN_FACES_DIR / filename
        logger.info(f"Saving to: {file_path}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        # Verify file was saved
        if file_path.exists():
            logger.info(f"✅ File saved successfully, size: {file_path.stat().st_size} bytes")
        else:
            logger.error(f"❌ File does not exist after save!")
        
        # Try to analyze face for age (optional, doesn't affect saving)
        if DEEPFACE_AVAILABLE:
            try:
                analysis = DeepFace.analyze(img_path=str(file_path), 
                                           actions=['age', 'gender', 'emotion'],
                                           enforce_detection=False,
                                           silent=True)
                logger.info(f"Face analysis - Estimated age: {analysis[0]['age']}")
            except Exception as e:
                logger.warning(f"Age analysis failed: {e}")
        
        return {
            "success": True,
            "message": "Face added successfully",
            "filename": filename
        }
    
    except Exception as e:
        logger.error(f"❌ Error in add-face-base64: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search for a face from base64 image with maximum tolerance for variations including age changes"""
    try:
        image_base64 = data.get('image')
        
        logger.info(f"🔍 Search request received, image length: {len(image_base64) if image_base64 else 0}")
        
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
        logger.info(f"Known faces count: {len(known_faces)}")
        
        if len(known_faces) == 0:
            os.remove(temp_path)
            return {"matched": False, "message": "No faces in database"}
        
        # Try to estimate age from query photo for better matching
        query_estimated_age = None
        try:
            analysis = DeepFace.analyze(img_path=str(temp_path), 
                                       actions=['age'],
                                       enforce_detection=False,
                                       silent=True)
            query_estimated_age = analysis[0]['age']
            logger.info(f"Query photo estimated age: {query_estimated_age}")
        except:
            logger.warning("Could not estimate age from query photo")
        
        best_match = None
        best_score = 0
        best_distance = float('inf')
        best_model_used = ""
        best_match_info = {}
        all_matches = []
        
        # ============================================
        # AGE-INVARIANT RECOGNITION CONFIGURATION
        # ============================================
        # Priority 1: ArcFace - Best for age-invariant matching
        # Priority 2: Facenet512 - Good all-rounder
        # Priority 3: VGG-Face - Traditional approach
        # Priority 4: Dlib - Very tolerant with low threshold
        # ============================================
        
        models_config = [
            # Model Name, Distance Metric, Threshold, Priority
            ("ArcFace", "cosine", 0.68, 1),        # Best for age changes
            ("ArcFace", "euclidean_l2", 1.13, 1),   # ArcFace with L2
            ("Facenet512", "cosine", 0.40, 2),      # Good general model
            ("Facenet512", "euclidean_l2", 1.04, 2), # L2 normalized
            ("VGG-Face", "cosine", 0.68, 3),         # Traditional
            ("Dlib", "cosine", 0.07, 4),            # Very tolerant
        ]
        
        # Multiple detector backends for better face detection
        detector_backends = [
            "retinaface",   # Best accuracy, handles small faces
            "mtcnn",        # Good alignment
            "dlib",         # Reliable
            "opencv",       # Fast fallback
        ]
        
        # Try each detector with each model configuration
        for detector in detector_backends:
            for model_name, distance_metric, threshold_val, priority in models_config:
                try:
                    logger.info(f"🔄 Trying detector: {detector}, model: {model_name}, metric: {distance_metric}")
                    
                    result = DeepFace.find(
                        img_path=str(temp_path),
                        db_path=str(KNOWN_FACES_DIR),
                        model_name=model_name,
                        distance_metric=distance_metric,
                        detector_backend=detector,
                        enforce_detection=False,
                        silent=True,
                        threshold=threshold_val,
                        normalization="base",
                        align=True  # Critical for pose variations
                    )
                    
                    if len(result) > 0 and not result[0].empty:
                        # Get the best match from this combination
                        match = result[0].iloc[0]
                        distance = float(match['distance'])
                        
                        # Convert distance to similarity score
                        if distance_metric == "cosine":
                            similarity = (1 - distance) * 100
                        elif distance_metric == "euclidean_l2":
                            similarity = max(0, min(100, 100 - (distance * 90)))
                        else:
                            similarity = max(0, min(100, 100 - (distance / 10)))
                        
                        logger.info(f"✅ Match found! {model_name}: similarity={similarity:.2f}%")
                        
                        # Get database photo path
                        db_photo_path = Path(match['identity'])
                        
                        # Extract metadata from filename
                        filename_parts = db_photo_path.name.split('_')
                        metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
                        metadata_parts = metadata_str.split('|')
                        
                        # Parse metadata
                        person_info = {
                            "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                            "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                            "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                            "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                            "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                            "photo_date": metadata_parts[5] if len(metadata_parts) > 5 else "Unknown"
                        }
                        
                        # Calculate age difference if available
                        age_diff = None
                        if person_info['age'] and query_estimated_age:
                            try:
                                db_age = float(person_info['age'])
                                age_diff = abs(db_age - query_estimated_age)
                                logger.info(f"Age difference: {age_diff:.1f} years")
                            except:
                                pass
                        
                        # Adjust similarity based on age difference
                        adjusted_similarity = similarity
                        if age_diff and age_diff > 3:
                            # If age difference > 3 years, boost confidence slightly
                            # because we expect some changes
                            adjusted_similarity = min(100, similarity * 1.1)
                            logger.info(f"Age difference detected, adjusted similarity: {adjusted_similarity:.2f}%")
                        
                        match_info = {
                            "similarity": adjusted_similarity,
                            "person_info": person_info,
                            "model": model_name,
                            "detector": detector,
                            "distance": distance,
                            "age_diff": age_diff
                        }
                        
                        all_matches.append(match_info)
                        
                        if adjusted_similarity > best_score:
                            best_score = adjusted_similarity
                            best_distance = distance
                            best_match = match
                            best_match_info = match_info
                            best_model_used = f"{detector}/{model_name}"
                            
                except Exception as e:
                    logger.warning(f"Combination {detector}/{model_name} failed: {e}")
                    continue
        
        # If we got multiple matches, do a second pass with ensemble voting
        if len(all_matches) >= 3:
            logger.info("Multiple matches found, performing ensemble voting...")
            
            # Group matches by person (identity)
            person_votes = {}
            for match_info in all_matches:
                person_name = match_info['person_info']['name']
                if person_name not in person_votes:
                    person_votes[person_name] = []
                person_votes[person_name].append(match_info['similarity'])
            
            # Calculate average similarity per person
            best_person = None
            best_avg_similarity = 0
            for person, similarities in person_votes.items():
                avg_sim = sum(similarities) / len(similarities)
                if avg_sim > best_avg_similarity:
                    best_avg_similarity = avg_sim
                    best_person = person
            
            logger.info(f"Ensemble voting result: {best_person} with avg similarity {best_avg_similarity:.2f}%")
            
            # If ensemble gives higher confidence, use it
            if best_avg_similarity > best_score:
                # Find the match info for this person
                for match_info in all_matches:
                    if match_info['person_info']['name'] == best_person:
                        best_score = best_avg_similarity
                        best_match_info = match_info
                        break
        
        os.remove(temp_path)
        
        if best_match_info:
            # Final match found
            person_info = best_match_info['person_info']
            
            # Add age progression note if age difference detected
            age_note = ""
            if best_match_info.get('age_diff') and best_match_info['age_diff'] > 2:
                age_note = f" (Age progressed by {best_match_info['age_diff']:.1f} years)"
            
            return {
                "matched": True,
                "identity": best_match_info['person_info']['name'],
                "person_info": person_info,
                "similarity": f"{best_score:.2f}%",
                "match_score": round(best_score, 2),
                "method": best_model_used,
                "age_difference": best_match_info.get('age_diff'),
                "message": f"✅ Match found with {best_score:.1f}% confidence{age_note}"
            }
        else:
            # Try manual embedding comparison as fallback
            logger.info("No match found with any combination, trying manual fallback...")
            try:
                # Get embedding of query face
                query_embedding = DeepFace.represent(
                    img_path=str(temp_path),
                    model_name="Facenet512",
                    detector_backend="retinaface",
                    enforce_detection=False,
                    align=True
                )[0]["embedding"]
                
                best_manual_match = None
                best_manual_score = 0
                best_manual_info = None
                
                # Compare with each known face manually
                for face_file in known_faces:
                    try:
                        target_embedding = DeepFace.represent(
                            img_path=str(face_file),
                            model_name="Facenet512",
                            detector_backend="retinaface",
                            enforce_detection=False,
                            align=True
                        )[0]["embedding"]
                        
                        # Calculate cosine similarity
                        q = np.array(query_embedding)
                        t = np.array(target_embedding)
                        similarity = np.dot(q, t) / (np.linalg.norm(q) * np.linalg.norm(t))
                        similarity_percent = similarity * 100
                        
                        if similarity_percent > 55:  # Lower threshold for fallback
                            # Extract metadata
                            filename_parts = face_file.name.split('_')
                            metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
                            metadata_parts = metadata_str.split('|')
                            
                            person_info = {
                                "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                                "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                                "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                                "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                                "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                                "photo_date": metadata_parts[5] if len(metadata_parts) > 5 else "Unknown"
                            }
                            
                            if similarity_percent > best_manual_score:
                                best_manual_score = similarity_percent
                                best_manual_match = face_file
                                best_manual_info = person_info
                                
                    except Exception as e:
                        continue
                
                if best_manual_match and best_manual_score > 55:
                    logger.info(f"✅ Manual fallback found match with {best_manual_score:.2f}%")
                    
                    return {
                        "matched": True,
                        "identity": best_manual_info['name'],
                        "person_info": best_manual_info,
                        "similarity": f"{best_manual_score:.2f}%",
                        "match_score": round(best_manual_score, 2),
                        "method": "manual/fallback",
                        "message": f"✅ Match found via fallback with {best_manual_score:.1f}% confidence"
                    }
            except Exception as e:
                logger.warning(f"Manual fallback failed: {e}")
            
            return {"matched": False, "message": "No match found with any method"}
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faces")
async def list_faces():
    """List all known faces with metadata including age information"""
    faces = []
    for f in KNOWN_FACES_DIR.glob("*.*"):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filename_parts = f.name.split('_')
            metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
            metadata_parts = metadata_str.split('|')
            
            face_info = {
                "filename": f.name,
                "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                "state": metadata_parts[4] if len(metadata_parts) > 4 else "",
                "photo_date": metadata_parts[5] if len(metadata_parts) > 5 else "Unknown",
                "size": f.stat().st_size
            }
            faces.append(face_info)
    
    return {"faces": faces, "count": len(faces)}

@app.delete("/clear-faces")
async def clear_all_faces():
    """Clear all faces from database (for testing)"""
    try:
        count = 0
        for f in KNOWN_FACES_DIR.glob("*.*"):
            f.unlink()
            count += 1
        return {"success": True, "message": f"Deleted {count} faces"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
