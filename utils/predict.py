import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
from PIL import Image

# Path dan URL model
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
model_path = os.path.join(model_dir, 'custom_trash_classifier.h5')
gdrive_file_id = '1XOmS1qu7OkmMfsHjBY6oBQ1mE9u73flp'
gdrive_download_url = f'https://drive.google.com/uc?export=download&id={gdrive_file_id}'

# Unduh model jika belum ada
def download_model_if_needed():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("📥 Model tidak ditemukan, mengunduh dari Google Drive...")
        try:
            response = requests.get(gdrive_download_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("✅ Model berhasil diunduh.")
            else:
                raise Exception(f"❌ Gagal mengunduh model: {response.status_code}")
        except Exception as e:
            print(f"❌ Error saat mengunduh model: {e}")
            raise RuntimeError("❌ Gagal mengunduh model.")

# Load model
download_model_if_needed()
print("📥 Loading model from:", model_path)
model = load_model(model_path)
print("✅ Model loaded!")

# Fungsi prediksi
def predict_image(img_path):
    print(f"📂 Memproses gambar: {img_path}")
    
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((128, 128))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        print(f"📊 Confidence: {confidence:.4f}")

        result = 'Anorganik' if confidence >= 0.46 else 'Organik'
        print(f"✅ Klasifikasi: {result}")
        return result, f"{confidence:.4f}"
    
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        raise RuntimeError(f"Model error: {e}")
