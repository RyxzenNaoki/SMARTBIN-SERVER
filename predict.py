import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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
model = load_model(model_path)
print("🧠 Model summary:")
model.summary()


# Kelas asli TrashNet
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Fungsi prediksi
def predict_image(img_path):
    print("📂 Memproses gambar:", img_path)

    try:
        # Sesuaikan dengan ukuran input model kamu!
        img = image.load_img(img_path, target_size=(128, 128))  # kalau model pakai RGB
    except Exception as e:
        raise ValueError(f"Gambar tidak valid atau rusak: {e}")
    
    img_array = image.img_to_array(img)              # (128, 128, 3)
    img_array = img_array / 255.0                     # normalisasi
    img_array = np.expand_dims(img_array, axis=0)    # (1, 128, 128, 3)

    print("📐 Shape final input model:", img_array.shape)

    try:
        prediction = model.predict(img_array)
        print("✅ Prediksi berhasil")
    except Exception as e:
        print("🔥 ERROR saat model.predict:", e)
        raise RuntimeError(f"Model error: {e}")

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    if predicted_class in ['paper', 'cardboard']:
        final_class = 'Organik'
    else:
        final_class = 'Anorganik'

    print(f"Hasil prediksi: {final_class} ({predicted_class})")
    return final_class, predicted_class
