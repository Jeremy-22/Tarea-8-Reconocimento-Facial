import tensorflow as tf
import datetime
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
import keras
from keras import losses
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop, Adam, Adamax

np.set_printoptions(precision=4)

batch_size = 16
img_height = 192
img_width = 192

#Cargamos los datos y cambios nuestras etiquetas con valor -1 a 0
df = pd.read_csv('/Users/jeryr/Codigos 2/Tarea-8-Reconocimento-Facial/attr_celeba_prepared.txt', sep=' ', header = None)
df = df.replace(-1,0)

files = tf.data.Dataset.from_tensor_slices(df[0])
attribute = tf.data.Dataset.from_tensor_slices(df.iloc[:, 1:11].to_numpy())
data = tf.data.Dataset.zip((files,attribute))

path_to_image = '/Users/jeryr/Codigos 2/img_align_celeba/img_align_celeba/'
def process_file(file_name, attribute):
    image = tf.io.read_file(path_to_image + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_width,img_height])
    return image, attribute

#Se crea un conjunto de datos de pares de image, attribute
#AUTOTUNE = tf.data.AUTOTUNE
#labeled_images = data.map(process_file,num_parallel_calls=AUTOTUNE)
labeled_images = data.map(process_file)

#Se divide el conjunto de datos en conjuntos de entrenamiento y validación
image_count = len(labeled_images)
image = labeled_images.shuffle(buffer_size=10*batch_size)
image = image.batch(batch_size)
train_img = image.take(int(0.8*image_count))
test_img = image.skip(int(0.8*image_count))
#val_size = int(image_count * 0.2)
#val_ds = labeled_images.take(val_size)

#print(tf.data.experimental.cardinality(train_ds).numpy())
#print(tf.data.experimental.cardinality(val_ds).numpy())

#def configure_for_performance(ds):
#  ds = ds.cache()
#  ds = ds.shuffle(buffer_size=1000)
#  ds = ds.batch(batch_size)
#  ds = ds.prefetch(buffer_size=AUTOTUNE)
#  return ds

#train_ds = configure_for_performance(train_ds)
#val_ds = configure_for_performance(val_ds)

#Construir y entrenar un modelo
num_classes = 10
learning_rate=0.005
epochs = 15
batch_size= 16
optimizer = Adam()
#Para cargar la red:
#modelo_cargado = tf.keras.models.load_model('Prueba_1.h5')
########################################
wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
#######################################

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(10, 3, input_shape = (img_width,img_height,3), activation='tanh',),
    tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  #tf.keras.layers.Conv2D(20, 3, activation='tanh'),
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #  tf.keras.layers.BatchNormalization(),
  #tf.keras.layers.Conv2D(10, 3, activation='relu'),
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
   # tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(5, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])


model.fit(
    train_img,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    validation_data=test_img,
    callbacks= [WandbCallback()])