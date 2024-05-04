
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
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,MaxPooling2D,Flatten
from keras.optimizers import RMSprop, Adam, Adamax, SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
batch_size = 128
img_height = 192
img_width = 192
num_classes = 10
learning_rate=0.0003
epochs = 20
optimizer = Adam()


train_dir = '/Users/jeryr/Codigos 2/co/co/dir/train_dat/' #directorio de entrenamiento
test_dir = '/Users/jeryr/Codigos 2/co/co/dir/test_dat/' #directorio de prueba


train_datagen = ImageDataGenerator(  
    rescale=1. / 255,
    zoom_range=0.4,
    shear_range=0.2,
    width_shift_range=0.4,
    height_shift_range=0.4,
    rotation_range = 20,
    horizontal_flip=True,
    fill_mode='nearest')

train = train_datagen.flow_from_directory(  
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test = test_datagen.flow_from_directory(  
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#Para cargar la red pre_entrenada:
pre_trained_model = tf.keras.models.load_model('celebA17.h5')
########################################
wandb.init(project="reconocimiento facial2")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
#######################################

model = tf.keras.Sequential()
model.add(pre_trained_model.layers[0])
model.add(pre_trained_model.layers[1])
model.add(pre_trained_model.layers[2])
model.add(pre_trained_model.layers[3])
model.add(pre_trained_model.layers[4])
model.add(pre_trained_model.layers[5])
#model.add(pre_trained_model.layers[6])
model.add(pre_trained_model.layers[7])
model.add(pre_trained_model.layers[8])
#model.add(pre_trained_model.layers[9])
model.add(pre_trained_model.layers[10])
model.add(Dense(113, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.00003, l2=0.00003)))
model.add(Dense(1, activation='sigmoid'))

for layer in model.layers[:10]:
    layer.trainable = False

model.summary()

#log_dir="Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/
#Callbacks para guardar el mejor modelo y detener el entrenamiento prematuramente si no hay mejoras
checkpoint = ModelCheckpoint("celebA28.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.fit(
    train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=test,
    callbacks= [WandbCallback(), checkpoint, early_stopping])

#Para guardar el modelo en disco
model.save("celebA28.h5")