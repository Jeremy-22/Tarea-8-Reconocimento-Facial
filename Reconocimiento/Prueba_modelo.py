from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
model = load_model('model-best.h5')
# Ejemplo de preprocesamiento de datos de prueba
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
# Evaluación del modelo en los datos de prueba
loss, accuracy = model.evaluate(test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
# Obtener predicciones del modelo en los datos de prueba
predictions = model.predict(test)
print(predictions)