import tensorflow as tf
#import datetime
#import pathlib
#import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers, models, regularizers,losses
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop, Adam

np.set_printoptions(precision=4)

batch_size = 128
img_height = 192
img_width = 192
num_classes = 1
learning_rate=1
epochs = 10
optimizer = Adam()

# Cargamos los datos y cambiamos nuestras etiquetas con valor -1 a 0
df = pd.read_csv('/Users/jeryr/Codigos 2/Tarea-8-Reconocimento-Facial/attr_celeba_prepared.txt', sep=' ', header=None)
df = df.replace(-1, 1)

# Obtén el número correcto de imágenes antes de crear el dataset
image_count = df.shape[0]

files = tf.data.Dataset.from_tensor_slices(df[0])
attribute = tf.data.Dataset.from_tensor_slices(df.iloc[:, 1:11].to_numpy())
data = tf.data.Dataset.zip((files, attribute))

path_to_image = '/Users/jeryr/Codigos 2/img_align_celeba/img_align_celeba/'
def process_file(file_name, attribute):
    image = tf.io.read_file(path_to_image + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_width, img_height])
    return image, attribute

labeled_images = data.map(process_file)

# Se divide el conjunto de datos en conjuntos de entrenamiento y validación
train_size = int(0.8 * image_count)
test_size = image_count - train_size

train_dataset = labeled_images.take(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)
test_dataset = labeled_images.skip(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)

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
  tf.keras.layers.Conv2D(16, 3, input_shape = (192,192,3), activation='relu',), #10, 8
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),#20, 16
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),#30,32
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),#10,64
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L1(l1=learning_rate)),#512
  #tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=regularizers.L1L2(l1=learning_rate, l2=learning_rate)),#20
  tf.keras.layers.Dropout(0.35257836296494716),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
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
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes, activation='relu')
])

# Corrección de las capas BatchNormalization
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(20, 3, input_shape=(img_width, img_height, 3), activation='tanh'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(20, 3, activation='tanh'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(30, 3, activation='tanh'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')  # Cambio en el número de neuronas y activación
])

# Optimizador: usar el optimizador definido anteriormente
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#checkpoint = ModelCheckpoint("celebA15.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Entrenamiento del modelo
model.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=test_dataset,
    callbacks=[WandbCallback()])#, checkpoint, early_stopping])
'''
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
# Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrena el modelo
model.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=test_dataset,
    callbacks=[WandbCallback()])
model.save("celebA05.h5")
