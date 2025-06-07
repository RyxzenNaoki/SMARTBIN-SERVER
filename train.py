import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Path ke dataset
base_dir = "./dataset"
img_size = 128
batch_size = 16
epochs = 10

# Data augmentation dan preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Arsitektur model CNN sederhana
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # 1 output: organik (0) atau anorganik (1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Simpan model terbaik selama training
checkpoint = ModelCheckpoint("custom_trash_classifier.h5", save_best_only=True, monitor='val_accuracy', mode='max')

# Latih model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint]
)

print("✅ Model selesai dilatih dan disimpan ke custom_trash_classifier.h5")
