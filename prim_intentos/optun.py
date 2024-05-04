import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbCallback
from keras import layers, models, regularizers
from keras.optimizers import Adam
import optuna

np.set_printoptions(precision=4)

batch_size = 160
img_height = 192
img_width = 192
num_classes = 10
epochs = 10

# Carga de datos
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

# División de datos en entrenamiento y prueba
train_size = int(0.8 * image_count)
test_size = image_count - train_size

train_dataset = labeled_images.take(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)
test_dataset = labeled_images.skip(train_size).shuffle(buffer_size=10*batch_size).batch(batch_size)

# Función objetivo para Optuna
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.7)

    # Construir el modelo con los hiperparámetros sugeridos por Optuna
    model = models.Sequential([
        layers.Conv2D(16, 3, input_shape=(img_height, img_width, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L1(l1=learning_rate)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compilar el modelo
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=test_dataset
    )

    # Devolver la precisión en el conjunto de validación para que Optuna pueda optimizarla
    return history.history['val_accuracy'][-1]

# Inicializar Weights and Biases
wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size

# Crear un estudio Optuna y ejecutar la optimización
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Obtener los mejores hiperparámetros encontrados por Optuna
best_params = study.best_params
print("Best params:", best_params)

# Construir el modelo final con los mejores hiperparámetros encontrados
best_model = models.Sequential([
    layers.Conv2D(16, 3, input_shape=(img_height, img_width, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L1(l1=best_params['learning_rate'])),
    layers.Dropout(best_params['dropout_rate']),
    layers.Dense(num_classes, activation='sigmoid')
])

best_model.compile(loss='binary_crossentropy',
                   optimizer=Adam(learning_rate=best_params['learning_rate']),
                   metrics=['accuracy'])

# Entrenar el modelo final con los mejores hiperparámetros en todos los datos
best_model.fit(
    labeled_images.batch(batch_size),
    epochs=epochs,
    verbose=1,
    callbacks=[WandbCallback()]
)

# Guardar el modelo final
best_model.save("celebA_optuna.h5")
