
import cv2
import os
import shutil

input_dir = "dir/Nueva/"

output_dir = "dir/train_dat/"

target_size = (192, 192)

for filename in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, filename))
    
    resized_img = cv2.resize(img, target_size)
    cv2.imwrite(os.path.join(output_dir, filename), resized_img)


# Directorios de origen y destino
origen_dir = "/Codigos 2/co/co/dir/dir/"  
train_dir = "/Codigos 2/co/co/dir/train_d/"   
test_dir = "/Codigos 2/co/co/dir/test_d/"     

# Creamos los directorios de entrenamiento y prueba si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Subdirectorio para imágenes de entrenamiento
train_subdir = os.path.join(train_dir, "rostros")
os.makedirs(train_subdir, exist_ok=True)

# Subdirectorio para imágenes de prueba
test_subdir = os.path.join(test_dir, "rostros")
os.makedirs(test_subdir, exist_ok=True)

# Copiamos todas las imágenes al subdirectorio de entrenamiento
for filename in os.listdir(origen_dir):
    filepath = os.path.join(origen_dir, filename)
    if os.path.isfile(filepath):
        shutil.copy(filepath, train_subdir)

# También copiamos las imágenes al subdirectorio de prueba
# En este ejemplo, simplemente copiamos las mismas imágenes que en el directorio de entrenamiento
# Puedes modificar esta parte según sea necesario para tu caso de uso
for filename in os.listdir(train_subdir):
    filepath = os.path.join(train_subdir, filename)
    if os.path.isfile(filepath):
        shutil.copy(filepath, test_subdir)

print("Estructura de directorios creada con éxito.")
import random

# Directorio donde están almacenadas todas tus imágenes
directorio_imagenes = '/Codigos 2/co/co/dir/train_d/rostros/'

# Directorios para el conjunto de entrenamiento y el conjunto de prueba
directorio_entrenamiento = '/Codigos 2/co/co/dir/train_dat/rostros/'
directorio_prueba = '/Codigos 2/co/co/dir/test_dat/rostros/'

# Crear directorios de entrenamiento y prueba si no existen
os.makedirs(directorio_entrenamiento, exist_ok=True)
os.makedirs(directorio_prueba, exist_ok=True)

# Obtener la lista de todas las imágenes en el directorio
imagenes = os.listdir(directorio_imagenes)

# Calcular el número de imágenes para el conjunto de entrenamiento y prueba
num_entrenamiento = int(len(imagenes) * 0.75)
num_prueba = len(imagenes) - num_entrenamiento

# Mezclar las imágenes para que la selección sea aleatoria
random.shuffle(imagenes)

# Copiar las primeras num_entrenamiento imágenes al directorio de entrenamiento
for imagen in imagenes[:num_entrenamiento]:
    origen = os.path.join(directorio_imagenes, imagen)
    destino = os.path.join(directorio_entrenamiento, imagen)
    shutil.copy(origen, destino)

# Copiar las imágenes restantes al directorio de prueba
for imagen in imagenes[num_entrenamiento:]:
    origen = os.path.join(directorio_imagenes, imagen)
    destino = os.path.join(directorio_prueba, imagen)
    shutil.copy(origen, destino)