import tensorflow as tf
import datetime
import pathlib
import os
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import wandb
from wandb.keras import WandbCallback
from tensorflow import keras
from keras import losses
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop, Adam, Adamax

np.set_printoptions(precision=4)
batch_size = 32
df = pd.read_csv('/Users/jeryr/Codigos 2/Tarea-8-Reconocimento-Facial/attr_celeba_prepared.txt', sep=' ', header = None)
df =df.replace(-1,0)

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))

path_to_images = '/Users/jeryr/Codigos 2/img_align_celeba/img_align_celeba/'

def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  #
    return image, attributes

#labeled_images = data.map(process_file)
AUTOTUNE = tf.data.AUTOTUNE
labeled_images = data.map(process_file,num_parallel_calls=AUTOTUNE)
#Se divide el conjunto de datos en conjuntos de entrenamiento y validación
image_count = len(labeled_images)
val_size = int(image_count * 0.2)
train_ds = labeled_images.skip(val_size)
val_ds = labeled_images.take(val_size)

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

num_classes = 10
learning_rate=0.005
epochs = 20
optimizer = Adam()
#Para cargar la red:
#modelo_cargado = tf.keras.models.load_model('Prueba_1.h5')
########################################
#wandb.init(project="reconocimiento facial")
#wandb.config.epochs = epochs
#wandb.config.batch_size = batch_size
#wandb.config.optimizer = optimizer
#######################################

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(50, 4, input_shape = (192,192,3), activation='relu',),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(80, 4, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(50, 4, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(20, 4, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['binary_accuracy'])

#log_dir="Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/


wandb.init(project="reconocimiento facial", config={"learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size})

# Loop de entrenamiento
for epoch in range(epochs):
    for batch_data, batch_labels in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(batch_data, training=True)
        with tf.GradientTape() as tape:
            predictions = model(batch_data, training=True)
            loss = model.compute_loss(batch_data, batch_labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Registro de métricas en Wandb
        wandb.log({"train_loss": loss})

    # Validación
    val_losses = []
    for batch_data, batch_labels in val_ds:
        predictions = model(batch_data, training=False)
        val_loss = model.compiled_loss(batch_labels, predictions)
        val_losses.append(val_loss)
    
    # Calcular la pérdida media de validación y registrarla en Wandb
    mean_val_loss = np.mean(val_losses)
    wandb.log({"val_loss": mean_val_loss})

#model.fit(
#    train_ds,
#    batch_size=batch_size,
#    epochs=10,
 #   verbose=1,
 #   validation_data=val_ds,
  #  callbacks= [WandbCallback()])

#Para guardar el modelo en disco
#model.save('rfnn.h5')


