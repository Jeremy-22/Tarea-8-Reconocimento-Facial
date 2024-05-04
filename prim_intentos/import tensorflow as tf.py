import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
from keras import layers, models, regularizers,losses
from keras.models import Sequential, load_model
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop, Adam

np.set_printoptions(precision=4)

batch_size = 160
img_height = 192
img_width = 192
num_classes = 10
learning_rate=0.0001
epochs = 5
optimizer = Adam()

df = pd.read_csv('/Users/jeryr/Codigos 2/Tarea-8-Reconocimento-Facial/attr_celeba_prepared.txt', sep=' ', header=None)
df = df.replace(-1, 0)

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

train_size = int(0.8 * image_count)
test_size = image_count - train_size

train_dataset = labeled_images.take(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)
test_dataset = labeled_images.skip(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)

wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
modelo = load_model('celebA02.h5')
modelo.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=test_dataset,
    callbacks= [WandbCallback()])

modelo.save("celebA08.h5")


