import os
import shutil
import time
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import predict_image
from typing import Optional, Any

# Firebase global vars
firebase_admin: Optional[Any] = None
db: Optional[Any] = None
firestore: Optional[Any] = None
SERVER_TIMESTAMP: Optional[Any] = None
firebase_initialized = False

app = FastAPI(title="SmartBin Classification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Firebase Firestore
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore import transactional, SERVER_TIMESTAMP
    
    service_account = {
        "type": "service_account",
        "project_id": os.getenv("FB_SA_PROJECT_ID"),
        "private_key_id": os.getenv("FB_SA_PRIVATE_KEY_ID"),
        "private_key": (os.getenv("FB_SA_PRIVATE_KEY") or "").replace("\\n", "\n"),
        "client_email": os.getenv("FB_SA_CLIENT_EMAIL"),
        "client_id": os.getenv("FB_SA_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("FB_SA_CLIENT_CERT_URL")
    }

    cred = credentials.Certificate(service_account)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_initialized = True

    print("✅ Firebase berhasil diinisialisasi dari ENV")

except Exception as e:
    print("❌ Gagal inisialisasi Firebase:", e)
    transactional = None
    SERVER_TIMESTAMP = None
    
print("📦 ENV Vars Loaded:")
print("Project ID:", os.getenv("FB_SA_PROJECT_ID"))
print("PK ID:", os.getenv("FB_SA_PRIVATE_KEY_ID"))
print("Client Email:", os.getenv("FB_SA_CLIENT_EMAIL"))
print("PK length:", len(os.getenv("FB_SA_PRIVATE_KEY") or "None"))

def get_today_date_id():
    """Generate date ID for today in format YYYY-MM-DD"""
    return datetime.now().strftime("%Y-%m-%d")

# FIXED: Unified counter update function
def update_trash_counter(trash_type):
    """Update counter sampah berdasarkan jenis (Organik/Anorganik) dengan schema terpadu"""
    if not firebase_initialized or db is None or transactional is None or SERVER_TIMESTAMP is None:
        print("Firebase tidak tersedia untuk update counter")
        return
    
    try:
        today_id = get_today_date_id()
        timestamp = int(time.time() * 1000)
        
        # FIXED: Update both daily counter and main counter in one transaction
        @transactional
        def update_unified_counter(transaction):
            # Type assertion to help Pylance understand db is not None here
            assert db is not None
            
            # Update daily counter
            daily_counter_ref = db.collection("sampah_counter").document(today_id)
            daily_doc = daily_counter_ref.get()
            
            if daily_doc.exists:
                daily_data = daily_doc.to_dict()
                daily_total = daily_data.get('total', 0) + 1
                daily_organik = daily_data.get('organik', 0)
                daily_anorganik = daily_data.get('anorganik', 0)
                
                if trash_type == 'Organik':
                    daily_organik += 1
                else:
                    daily_anorganik += 1
                
                transaction.update(daily_counter_ref, {
                    'total': daily_total,
                    'organik': daily_organik,
                    'anorganik': daily_anorganik,
                    'lastUpdate': timestamp,
                    'date': today_id
                })
            else:
                # Create new daily document
                daily_initial_data = {
                    'total': 1,
                    'organik': 1 if trash_type == 'Organik' else 0,
                    'anorganik': 1 if trash_type == 'Anorganik' else 0,
                    'date': today_id,
                    'lastUpdate': timestamp
                }
                transaction.set(daily_counter_ref, daily_initial_data)
            
            # FIXED: Update main counter with organik/anorganik fields
            main_counter_ref = db.collection("sampah").document("counter")
            main_doc = main_counter_ref.get()
            
            if main_doc.exists:
                main_data = main_doc.to_dict()
                main_total = main_data.get('jumlah', 0) + 1
                main_organik = main_data.get('organik', 0)
                main_anorganik = main_data.get('anorganik', 0)
                
                if trash_type == 'Organik':
                    main_organik += 1
                else:
                    main_anorganik += 1
                
                transaction.update(main_counter_ref, {
                    'jumlah': main_total,
                    'organik': main_organik,
                    'anorganik': main_anorganik,
                    'unit': 'items',
                    'lastUpdate': SERVER_TIMESTAMP
                })
            else:
                # Create new main counter document
                main_initial_data = {
                    'jumlah': 1,
                    'organik': 1 if trash_type == 'Organik' else 0,
                    'anorganik': 1 if trash_type == 'Anorganik' else 0,
                    'unit': 'items',
                    'lastUpdate': SERVER_TIMESTAMP
                }
                transaction.set(main_counter_ref, main_initial_data)
        
        # Execute transaction
        transaction = db.transaction()
        update_unified_counter(transaction)
        
        print(f"✅ Counter berhasil diupdate: {trash_type} (+1)")
        
    except Exception as e:
        print(f"❌ Error update counter: {e}")

@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    if file.filename is None:
        tmp_path = "uploaded_image.jpg"
    else:
        tmp_path = file.filename.replace(" ", "_")

    if not tmp_path:
        tmp_path = "uploaded_image.jpg"

    print(f"Menyimpan file sementara ke: {tmp_path}")

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error menyimpan file: {e}")

    try:
        result, original_class = predict_image(tmp_path)
        print(f"Hasil klasifikasi: {result} (asli: {original_class})")

        if firebase_initialized and db is not None:
            try:
                timestamp = int(time.time() * 1000)

                # Simpan hasil klasifikasi terakhir
                db.collection("klasifikasi").document("terakhir").set({
                    "jenis": result,
                    "kelas_asli": original_class,
                    "timestamp": timestamp
                })

                # Tambahkan ke riwayat klasifikasi
                db.collection("riwayat_klasifikasi").add({
                    "jenis": result,
                    "kelas_asli": original_class,
                    "timestamp": timestamp
                })

                # Status untuk ESP32
                db.collection("status_sampah").document("esp32").set({
                    "jenis": result,
                    "perlu_dibuka": True,
                    "timestamp": timestamp
                })

                # FIXED: Update counter sampah berdasarkan jenis dengan schema terpadu
                update_trash_counter(result)

                print("Hasil berhasil disimpan ke Firestore")

            except Exception as e:
                print(f"Error saat menyimpan ke Firestore: {e}")
        else:
            print("Hasil tidak disimpan ke Firestore (dinonaktifkan)")
    except Exception as e:
        print(f"Error saat klasifikasi: {e}")
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"File sementara {tmp_path} berhasil dihapus")
            except Exception as e:
                print(f"Error saat menghapus file sementara: {e}")

    return {"jenis": result, "kelas_asli": original_class}

@app.get("/status")
async def status():
    firebase_status = "aktif" if firebase_initialized else "nonaktif"
    return {
        "status": "online",
        "message": f"API server berjalan (Firestore: {firebase_status})"
    }

@app.get("/counter/today")
async def get_today_counter():
    """Get counter sampah hari ini"""
    if not firebase_initialized or db is None:
        raise HTTPException(status_code=500, detail="Firestore tidak tersedia")

    try:
        today_id = get_today_date_id()
        doc_ref = db.collection("sampah_counter").document(today_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            return {
                "total": 0,
                "organik": 0,
                "anorganik": 0,
                "date": today_id,
                "lastUpdate": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FIXED: Add endpoint to get main counter with organik/anorganik breakdown
@app.get("/counter/main")
async def get_main_counter():
    """Get main counter with organik/anorganik breakdown"""
    if not firebase_initialized or db is None:
        raise HTTPException(status_code=500, detail="Firestore tidak tersedia")

    try:
        doc_ref = db.collection("sampah").document("counter")
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            return {
                "jumlah": data.get("jumlah", 0),
                "organik": data.get("organik", 0),
                "anorganik": data.get("anorganik", 0),
                "unit": data.get("unit", "items"),
                "lastUpdate": data.get("lastUpdate")
            }
        else:
            return {
                "jumlah": 0,
                "organik": 0,
                "anorganik": 0,
                "unit": "items",
                "lastUpdate": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/status_sampah/esp32")
async def get_status():
    if not firebase_initialized or db is None:
        raise HTTPException(status_code=500, detail="Firestore tidak tersedia")

    try:
        doc_ref = db.collection("status_sampah").document("esp32")
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return {"error": "Dokumen tidak ditemukan"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_status():
    if not firebase_initialized or db is None:
        raise HTTPException(status_code=500, detail="Firestore tidak tersedia")

    try:
        db.collection("status_sampah").document("esp32").update({
            "perlu_dibuka": False
        })
        return {"message": "Status berhasil direset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FIXED: Add endpoint to reset counters (for testing)
@app.post("/reset-counters")
async def reset_counters():
    """Reset all counters to zero (for testing)"""
    if not firebase_initialized or db is None:
        raise HTTPException(status_code=500, detail="Firestore tidak tersedia")

    try:
        # Reset main counter
        db.collection("sampah").document("counter").set({
            'jumlah': 0,
            'organik': 0,
            'anorganik': 0,
            'unit': 'items',
            'lastUpdate': SERVER_TIMESTAMP
        })
        
        # Reset today's counter
        today_id = get_today_date_id()
        db.collection("sampah_counter").document(today_id).set({
            'total': 0,
            'organik': 0,
            'anorganik': 0,
            'date': today_id,
            'lastUpdate': int(time.time() * 1000)
        })
        
        return {"message": "Semua counter berhasil direset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model-info")
async def model_info():
    from utils.predict import model
    return {
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "model_name": model.name if hasattr(model, 'name') else "unknown"
    }

if __name__ == "__main__":
    import uvicorn
    print("Memulai server API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)