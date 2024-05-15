import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "#path of the training dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    "#path of the validation dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // 32,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // 32)



# Loading and preprocessing the image
test_img_path = '#path off the test image'
test_img = cv2.imread(test_img_path)
if test_img is not None:
    test_img = cv2.resize(test_img, (150, 150))
    #normalize 
    test_input = np.expand_dims(test_img, axis=0) / 255.0 
    prediction = model.predict(test_input)
    if prediction[0][0] < 0.5:
        print("The person is happy.")
    else:
        print("The person is sad.")
else:
    print("Error: The image could not be loaded. Please check the path.")
