import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense

np.set_printoptions(precision=4)

batch_size = 160
img_height = 192
img_width = 192
num_classes = 10
learning_rate=0.00001
epochs = 4
optimizer = Adam()

# Cargamos los datos y cambiamos nuestras etiquetas con valor -1 a 0
df = pd.read_csv('attr_celeba_prepared.txt', sep=' ', header=None)
df = df.replace(-1, 0)

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
print(train_size)
print(test_size)
train_dataset = labeled_images.take(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)
test_dataset = labeled_images.skip(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)

wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
model1 = load_model('celebA19.h5')


model1.summary()

#0.33783----14____lernin=190
# Callbacks para guardar el mejor modelo y detener el entrenamiento prematuramente si no hay mejoras
checkpoint = ModelCheckpoint("celebA20.h5", monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=2, restore_best_weights=True)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model1.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=test_dataset,
    callbacks=[WandbCallback(), checkpoint, early_stopping])

model1.save("celebA20.h5")
