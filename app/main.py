@app.post("/search-base64")
async def search_face_base64(data: dict):
    """Search for a face from base64 image with maximum tolerance for variations"""
    try:
        image_base64 = data.get('image')
        
        logger.info(f"Search request received, image length: {len(image_base64) if image_base64 else 0}")
        
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
        
        best_match = None
        best_score = 0
        best_distance = float('inf')
        best_model_used = ""
        
        # ============================================
        # VARIATION 1: MULTIPLE MODELS
        # Har model ki apni strength hai
        # ============================================
        models_to_try = [
            # Model Name, Distance Metric, Threshold
            ("Facenet512", "cosine", 0.30),      # Best overall accuracy [citation:8]
            ("Facenet512", "euclidean_l2", 1.04), # L2 normalized more stable [citation:10]
            ("Facenet512", "euclidean", 23.56),   # Higher threshold for variations
            ("ArcFace", "cosine", 0.68),          # Excellent for angle variations [citation:2]
            ("ArcFace", "euclidean_l2", 1.13),    # ArcFace with L2 normalization
            ("VGG-Face", "cosine", 0.68),         # Good all-rounder
            ("VGG-Face", "euclidean_l2", 1.17),   # VGG-Face with L2
            ("Dlib", "cosine", 0.07),             # Very low threshold - very tolerant
            ("SFace", "cosine", 0.593),           # Robust to lighting changes
        ]
        
        # ============================================
        # VARIATION 2: MULTIPLE DETECTOR BACKENDS
        # Better face detection = better recognition
        # ============================================
        detector_backends = [
            "retinaface",  # Best accuracy, handles small faces, landmarks [citation:8]
            "mtcnn",       # Very good alignment [citation:4]
            "dlib",        # Reliable 
            "opencv",      # Fast fallback
        ]
        
        # Try each detector with each model configuration
        for detector in detector_backends:
            for model_name, distance_metric, threshold_val in models_to_try:
                try:
                    logger.info(f"Trying detector: {detector}, model: {model_name}, metric: {distance_metric}")
                    
                    # ============================================
                    # VARIATION 3: FACE ALIGNMENT
                    # Align faces to handle head tilt/rotation
                    # ============================================
                    result = DeepFace.find(
                        img_path=str(temp_path),
                        db_path=str(KNOWN_FACES_DIR),
                        model_name=model_name,
                        distance_metric=distance_metric,
                        detector_backend=detector,  # Try different detectors
                        enforce_detection=False,
                        silent=True,
                        threshold=threshold_val,
                        normalization="base",  # Normalize lighting variations
                        align=True  # CRITICAL: Align faces for pose variations
                    )
                    
                    if len(result) > 0 and not result[0].empty:
                        match = result[0].iloc[0]
                        distance = float(match['distance'])
                        
                        # Convert distance to similarity score based on metric
                        if distance_metric == "cosine":
                            similarity = (1 - distance) * 100
                        elif distance_metric == "euclidean_l2":
                            similarity = max(0, min(100, 100 - (distance * 100)))
                        else:  # euclidean
                            similarity = max(0, min(100, 100 - (distance / 10)))
                        
                        logger.info(f"✅ Match found! Detector: {detector}, Model: {model_name}, Similarity: {similarity:.2f}%")
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_distance = distance
                            best_match = match
                            best_model_used = f"{detector}/{model_name}"
                            
                except Exception as e:
                    logger.warning(f"Combination {detector}/{model_name} failed: {e}")
                    continue
        
        os.remove(temp_path)
        
        if best_match is not None:
            identity_path = Path(best_match['identity'])
            
            # Extract metadata from filename
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
            
            return {
                "matched": True,
                "identity": identity_path.name,
                "person_info": person_info,
                "similarity": f"{best_score:.2f}%",
                "match_score": round(best_score, 2),
                "method": best_model_used,
                "message": f"✅ Match found with {best_score:.1f}% confidence"
            }
        else:
            logger.info("No match found with any combination")
            
            # ============================================
            # VARIATION 4: FALLBACK TO REPRESENT + CUSTOM COMPARISON
            # Last resort - compare embeddings manually
            # ============================================
            try:
                logger.info("Trying manual embedding comparison as fallback...")
                
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
                        
                        # Calculate cosine similarity manually
                        import numpy as np
                        q = np.array(query_embedding)
                        t = np.array(target_embedding)
                        similarity = np.dot(q, t) / (np.linalg.norm(q) * np.linalg.norm(t))
                        similarity_percent = similarity * 100
                        
                        if similarity_percent > 60 and similarity_percent > best_manual_score:
                            best_manual_score = similarity_percent
                            best_manual_match = face_file
                            
                    except:
                        continue
                
                if best_manual_match and best_manual_score > 60:
                    # Manual match found
                    filename_parts = best_manual_match.name.split('_')
                    metadata_str = filename_parts[0] if len(filename_parts) > 0 else ""
                    metadata_parts = metadata_str.split('|')
                    
                    person_info = {
                        "name": metadata_parts[0] if len(metadata_parts) > 0 else "Unknown",
                        "age": metadata_parts[1] if len(metadata_parts) > 1 else "",
                        "mobile": metadata_parts[2] if len(metadata_parts) > 2 else "",
                        "city": metadata_parts[3] if len(metadata_parts) > 3 else "",
                        "state": metadata_parts[4] if len(metadata_parts) > 4 else ""
                    }
                    
                    return {
                        "matched": True,
                        "identity": best_manual_match.name,
                        "person_info": person_info,
                        "similarity": f"{best_manual_score:.2f}%",
                        "match_score": round(best_manual_score, 2),
                        "method": "manual/fallback",
                        "message": f"✅ Match found via fallback with {best_manual_score:.1f}% confidence"
                    }
            except Exception as e:
                logger.warning(f"Manual fallback failed: {e}")
            
            return {"matched": False, "message": "No match found with any method"}
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
