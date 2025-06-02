import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
from PIL import Image

# Path ke model lokal setelah diunduh
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
model_path = os.path.join(model_dir, 'trashnet_model.h5')

# URL Google Drive (file ID)
gdrive_file_id = '1CtFIBw05q63ptPKohRSjaxJrxjxtudmT'
gdrive_download_url = f'https://drive.google.com/uc?export=download&id={gdrive_file_id}'

# Unduh model jika belum tersedia
def download_model_if_needed():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("Model tidak ditemukan, mengunduh dari Google Drive...")
        try:
            response = requests.get(gdrive_download_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("Model berhasil diunduh.")
            else:
                raise Exception(f"Gagal mengunduh model: {response.status_code}")
        except Exception as e:
            print(f"Error saat mengunduh model: {e}")
            raise RuntimeError("Gagal mengunduh model.")

# Load model
download_model_if_needed()
print("📥 Loading model from:", model_path)
model = load_model(model_path)
print("✅ Model loaded successfully!")
print("🧠 Model summary:")
model.summary()
print("🔍 Model input shape:", model.input_shape)
print("🔍 Model output shape:", model.output_shape)

# Kelas asli TrashNet
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Keywords untuk klasifikasi organik yang lebih komprehensif
organic_keywords = [
    'paper', 'cardboard',  # Kertas dan kardus (biodegradable)
    'food', 'leaf', 'leaves', 'makanan', 'daun',  # Makanan dan daun
    'organic', 'bio', 'compost', 'organik',  # Keywords organik umum
    'wood', 'kayu', 'ranting', 'branch',  # Material kayu
    'fruit', 'vegetable', 'buah', 'sayur',  # Buah dan sayuran
    'tissue', 'napkin', 'tisu'  # Tissue paper
]

# FIXED: Keywords untuk klasifikasi anorganik yang lebih komprehensif
anorganic_keywords = [
    'glass', 'metal', 'plastic',  # Material anorganik utama
    'bottle', 'can', 'aluminium', 'aluminum',  # Wadah logam/kaca
    'steel', 'iron', 'copper', 'brass',  # Jenis logam
    'polyethylene', 'polystyrene', 'pvc', 'pet',  # Jenis plastik
    'synthetic', 'artificial', 'sintetis',  # Material sintetis
    'electronic', 'elektronik', 'battery', 'baterai',  # Elektronik
    'ceramic', 'keramik', 'porcelain',  # Keramik
    'rubber', 'karet', 'foam', 'busa',  # Karet dan busa
    'fabric', 'cloth', 'textile', 'kain',  # Tekstil sintetis
    'wire', 'cable', 'kabel', 'kawat'  # Kabel dan kawat
]

# FIXED: Fungsi untuk mengecek apakah suatu item termasuk organik dengan logika yang diperbaiki
def is_organic(predicted_class, confidence_threshold=0.3):
    """
    Menentukan apakah sampah termasuk organik berdasarkan:
    1. Kelas prediksi model
    2. Keywords organik dan anorganik
    3. Confidence threshold
    4. Prioritas anorganik untuk material yang jelas anorganik
    """
    predicted_lower = predicted_class.lower()
    
    # FIXED: Cek anorganik keywords terlebih dahulu (prioritas tinggi)
    for keyword in anorganic_keywords:
        if keyword in predicted_lower:
            print(f"🔍 Found anorganic keyword: '{keyword}' in '{predicted_class}'")
            return False
    
    # Klasifikasi berdasarkan kelas TrashNet yang jelas anorganik
    if predicted_class in ['glass', 'metal', 'plastic']:
        print(f"🔍 Class '{predicted_class}' is clearly anorganic")
        return False
    
    # Klasifikasi berdasarkan kelas TrashNet yang jelas organik
    if predicted_class in ['paper', 'cardboard']:
        print(f"🔍 Class '{predicted_class}' is clearly organic")
        return True
    
    # Cek keywords organik
    for keyword in organic_keywords:
        if keyword in predicted_lower:
            print(f"🔍 Found organic keyword: '{keyword}' in '{predicted_class}'")
            return True
    
    # FIXED: Trash bisa organik atau anorganik, default ke anorganik jika tidak ada keyword organik
    if predicted_class == 'trash':
        # Cek apakah ada keyword organik dalam konteks trash
        has_organic_keyword = any(keyword in predicted_lower for keyword in organic_keywords)
        print(f"🔍 Trash classification - organic keywords found: {has_organic_keyword}")
        return has_organic_keyword
    
    # FIXED: Default ke anorganik jika tidak ada klasifikasi yang jelas
    print(f"🔍 No clear classification for '{predicted_class}', defaulting to anorganic")
    return False

# Fungsi prediksi dengan preprocessing yang benar
def predict_image(img_path):
    print("📂 Memproses gambar:", img_path)

    try:
        # Buka gambar menggunakan PIL untuk handling yang lebih baik
        img = Image.open(img_path)
        
        # Convert ke RGB jika belum (untuk handle RGBA, Grayscale, dll)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"🔄 Converted image from {img.mode} to RGB")
        
        # Resize ke ukuran yang diharapkan model (128x128)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        print(f"📐 Resized to: {img.size}")
        
        # Convert ke numpy array
        img_array = np.array(img)
        print(f"🔍 Shape after conversion: {img_array.shape}")
        
        # Normalisasi pixel values ke range [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        print("📐 Shape final input model:", img_array.shape)
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        raise ValueError(f"Gambar tidak valid atau rusak: {e}")

    try:
        print("🔥 Melakukan prediksi...")
        prediction = model.predict(img_array)
        print("✅ Prediksi berhasil")
        print(f"📊 Prediction shape: {prediction.shape}")
        print(f"📊 Prediction values: {prediction[0]}")
        
    except Exception as e:
        print("🔥 ERROR saat model.predict:", e)
        raise RuntimeError(f"Model error: {e}")

    # Get class dengan confidence tertinggi
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])

    # FIXED: Mapping ke kategori akhir dengan logika yang diperbaiki
    print(f"🎯 Raw prediction: {predicted_class} (confidence: {confidence:.4f})")
    
    if is_organic(predicted_class, confidence):
        final_class = 'Organik'
        print(f"✅ Classification result: ORGANIC")
    else:
        final_class = 'Anorganik'
        print(f"✅ Classification result: ANORGANIC")

    print(f"✅ Final result: {final_class}")
    
    return final_class, predicted_class