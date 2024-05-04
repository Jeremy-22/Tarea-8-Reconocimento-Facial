import tensorflow as tf
#import datetime
#import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
import keras
from keras import layers, models, regularizers,losses
from keras.models import Sequential, load_model
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop, Adam

np.set_printoptions(precision=4)

batch_size = 80
img_height = 192
img_width = 192
num_classes = 10
learning_rate=0.0001
epochs = 15
optimizer = Adam()

# Cargamos los datos y cambiamos nuestras etiquetas con valor -1 a 0
df = pd.read_csv('attr_celeba_prepared.txt', sep=' ', header=None)
df = df.replace(-1, 1)

#obteenmos el numero de datos
image_count = df.shape[0]

files = tf.data.Dataset.from_tensor_slices(df[0])
attribute = tf.data.Dataset.from_tensor_slices(df.iloc[:, 1:11].to_numpy())
data = tf.data.Dataset.zip((files, attribute))

path_to_image = '/Users/jeryr/Codigos 2/img_align_celeba/img_align_celeba/'
def process_file(file_name, attribute):
    image = tf.io.read_file(path_to_image + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_width, img_height])
    image /= 255
    return image, attribute

#labeled_images = data.map(process_file)
#paralelisamos los datos
num_cpus = tf.data.AUTOTUNE
labeled_images = data.map(process_file, num_parallel_calls=num_cpus)

# Se divide el conjunto de datos en conjuntos de entrenamiento y validación
val_size = int(image_count * 0.2)
train_ds = labeled_images.skip(val_size)
val_ds = labeled_images.take(val_size)
'''
train_size = int(0.8 * image_count)
test_size = image_count - train_size
print(train_size)
print(train_ds)
train_dataset = labeled_images.take(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)
test_dataset = labeled_images.skip(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)
'''

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=num_cpus)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# Verificar las dimensiones de los lotes
#for images, labels in train_dataset.take(1):
#    print("Forma de los lotes de imágenes:", images.shape)
#    print("Forma de los lotes de etiquetas:", labels.shape)

########################################
wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
####################################### 
'''
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(50, 4, input_shape = (img_width,img_height,3), activation='relu',),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(80, 4, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(50, 4, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(20, 4, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(50, activation='relu'),  
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, input_shape = (img_height,img_width,3), activation='relu',), #10, 8
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(32, 3, activation='tanh'),#20, 16
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.BatchNormalization(),
  #tf.keras.layers.Conv2D(64, 3, activation='relu'),#30,32
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.BatchNormalization(),
  #tf.keras.layers.Conv2D(128, 3, activation='tanh'),#10,64
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1(l1=learning_rate)),#512
  #tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=regularizers.L1L2(l1=learning_rate, l2=learning_rate)),#20
  tf.keras.layers.Dropout(0.35257836296494716),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
'''
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, input_shape = (img_width,img_height,3), activation='relu',),
   # tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L1(l1=00.0003)),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=regularizers.L1(l1=00.0003)),  
  #tf.keras.layers.Dense(1, activation='sigmoid')
])

#model = load_model('celebA15.h5')
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


model.fit(
    train_ds,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=val_ds)
   # callbacks= [WandbCallback()])

model.save("celebA20.h5")
