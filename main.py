import os
import shutil
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import predict_image, show_model_summary

# Firebase global vars
firebase_admin = None
db = None
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
    
print("📦 ENV Vars Loaded:")
print("Project ID:", os.getenv("FB_SA_PROJECT_ID"))
print("PK ID:", os.getenv("FB_SA_PRIVATE_KEY_ID"))
print("Client Email:", os.getenv("FB_SA_CLIENT_EMAIL"))
print("PK length:", len(os.getenv("FB_SA_PRIVATE_KEY") or "None"))



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
    


@app.get("/model-summary")
def print_model_summary():
    show_model_summary()
    return {"message": "Model summary printed to logs"}




if __name__ == "__main__":
    import uvicorn
    print("Memulai server API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
