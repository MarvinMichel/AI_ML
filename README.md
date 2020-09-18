# Emojify
Machine Learning project for the course 'Artificial Intelligence', provided by the Amsterdam University of Applied Sciences. This product exists of a Deep Learning model and a GUI. The project contains the FER-2013 dataset and a trained model, made with train.py.

This app uses your webcam and facial recognition to check your facial expression. Based on your facial expression, the model will make a prediction of your emotional state and will print a matching Emoji with it. The app consists of a Covolutional Neural Network and a simple GUI.

#### Used modules
> To install these modules, check their documentation.
- [OpenCV](https://docs.opencv.org/master/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Keras](https://keras.io/)

#### Used dataset
- [FER-2013](https://www.kaggle.com/msambare/fer2013)

# How does it work?
The training model is a Convolutional Neural Network. It is an algorithm that uses several layers of filters and neurons working together to recognize abstract features. This makes it possible for our system to not just recognize a face, but look at your facial expression to see what emotion you are expressing. We start by importing every module that we'll need in the process.
```python
import numpy as np
import cv2
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
```
We'll use `Numpy` to change the dimensions from the images inside our arrays. For the image processing and datageneration we'll import `Keras`. Our Machine Learning platform will be `Tensorflow`. And last but not least we'll import `OpenCV` to use the webcam and generate a boundingbox around faces to recognize.

To train our model, we'll need data. For our project we used FER-2013 as our dataset. But you are free to use any data suitable for the task. Just make sure you fit the coding by your type of images and check you data for any potential biases. You can download the FER-2013 dataset [here](https://www.kaggle.com/msambare/fer2013). Import the directories with data into your project folder and call them within Python.
```python
train_dir = 'data/train'
val_dir = 'data/test'
```
We'll have two directories: one for test data and another for validation. The validation directory will be the 'test' folder. Don't get confused by the names. If the name doesn't suite you, feel free to change the directory names to something else. Just make sure it will be recognizable for someone else.

Next step, is to extract data from the images. This is where the `ImageDataGenerator` member of Keras will be our saviour! It will generate batches of tensor image data with real-time data augmentation. We'll call it's method and give it a rescale parameter of 1./255 to target an output of 0 and 1 instead of 0-255 from our images' RGB coefficients. It's possible to add different parameters. Check the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) for more info. But for now will stick with just one.
```python
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
```

Now it's time to do something with our datagenerators. Our goal is to use them to train and validate our data. By creating a generator for those tasks we'll do just that.
```python
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
```
We'll give them several parameters. The first is the directories that we created earlier, containing our data. We'll give it a `target_size`, which shows our model what size our data has. In our case, we use greyscale images sized 48px by 48px. The next parameter will be the color mode. This is import the clarify, since we'll be giving our trainingmodel an inputshape that will use a third parameter. If we do not specify the `color_mode`, it will give problem later on. The last parameter just tells our model to use classification and label our data.

Next will be our CNN model itself. It uses several type of layers. Don't worry if it scares you right now. We'll be walking trough them one by one.
```python
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])
```
As you can see we've got several `Conv2D` layers. These are our Convolutional Layers. They each take a set of filters, which is represented by the first number. The first Convolutional Layer takes an extra parameter with the input shape. This is the prevered shape of the images. The activation parameter gets set on 'relu'. This will apply a linear unit activation, providing us with either a 0 or a 1 as the threshold. The filter that matches the image the best gets a 1. All the others will be provided with a 0.

The other duplicate layers are `MaxPooling2D` and `Dropout`. The MaxPooling2D layers' name gives it away; it's a Maxpooling layer. This is a function that accumulate features from the picture, generated by the filters convolved over the image. This gives the image an abstracted form, and will help prevent overfitting our model. The Dropouts are a way of preventing overfitting as well. These layers will select random neurons during the training and will ignore them. This makes the model less sensitive to the specific weights of neurons and will ultimately generalize our model.
