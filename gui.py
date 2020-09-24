import tkinter as tk
from tkinter import *
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator 

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

#load weights  
model.load_weights('model.h5')  

emotion_dictionary={
  0: 'Angry',
  1: 'Disgusted',
  2: 'Fear',
  3: 'Happy',
  4: 'Neutral',
  5: 'Sad',
  6: 'Suprised'
}

# emoji_dictionary = {
#   0: 'images/angry.png',
#   1: 'images/disgusted.png',
#   2: 'images/fear.png',
#   3: 'images/happy.png',
#   4: 'images/neutral.png',
#   5: 'images/sad.png',
#   6: 'images/suprised.png'
# } 


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)  

while True:  
    frame, test_img = cap.read()
    if not frame:
        continue
    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = face_casc.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(test_img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        prediction = model.predict(img_pixels)

        #find max indexed array
        max_index = int(np.argmax(prediction))

        predicted_emotion = emotion_dictionary[max_index]

        cv2.putText(test_img, predicted_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Emojify', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows