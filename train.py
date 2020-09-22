import numpy as np
import cv2
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#Dataset directories
train_dir = 'data/train'
val_dir = 'data/test'

#Image Datagenerators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(48, 48),
  batch_size=64,
  color_mode='grayscale',
  class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
  val_dir,
  target_size=(48, 48),
  batch_size=64,
  color_mode='grayscale',
  class_mode='categorical'
)

# CNN Model
model=tf.keras.models.Sequential([
    #Ideal inputshape
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #Prevent overfitting
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    #Layers with 1024 neurons
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #Dense with posible classes
    tf.keras.layers.Dense(7, activation='softmax')
])

#Start webcam feed
cv2.ocl.setUseOpenCL(False)

#All posible classes
emotion_dictionary={
  0: 'Angry',
  1: 'Disgusted',
  2: 'Fear',
  3: 'Happy',
  4: 'Neutral',
  5: 'Sad',
  6: 'Suprised'
}

#Compile model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=1,
    validation_data = validation_generator,
    verbose = 1
)

model.save_weights("model.h5")

