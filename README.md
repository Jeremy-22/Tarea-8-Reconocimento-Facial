# Tarea-8-Reconocimento-Facial
![Top language](https://img.shields.io/badge/python-100%25-blue?style=for-the-badge
)

[![Last commit](https://img.shields.io/badge/LAST%20COMMIT-MAY%202024-orange?style=for-the-badge)](https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/commit/f0d02215bc75e64c5d5da2ccd7ac574dba98a616)
[![License](https://img.shields.io/badge/LICENSE-GNU-green?style=for-the-badge
)](https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/blob/main/LICENSE)

### Predicción de atributos de la base de datos [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (la cual contiene imágenes de rostros etiquetadas con 40 características o atributos).
---
Para realizar este modelo primero se procesaron los datos, en donde se quito el doble espacio de algunos datos, cargamos los datos y cambiamos nuestras etiquetas con valor -1 a 0, sé dividio el conjunto de datos entre validación y entrenamiento. Al realizar esto, se llamó a Wanbd.ia y se contruyo la arquitectura del modelo con capas Convolucionales 2D, con un clasificador de capas densas, y usando:
```bash
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
```
<center>
  <h4>Primer intento:</h4>
  <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial/runs/388f5ky0?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png" alt="Open In Colab"/>
  </a>
</center>
<br>
<center>
 <h4>Predicción de atributos:</h4>
 <a href="https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/blob/main/atributos/atributos_intentos.py" target="_blank">
    <img width="20%" src="https://programacion.net/files/article/20160614050620_python-logo.png"/>
  </a>
</center>

Para obtener una mejor precisión opte por seguir entrenando la red, cambiando el batch_size, el learning_rate, aumentar capas a la arquitectura del modelo y regularizadores, pero con esto solo pude tener una precisión maxima del 34.66%.
<center>
  <h4>Maxima precision con accuracy:</h4>
  <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png" alt="Open In Colab"/>
  </a>
</center>
<br>

Por lo que, decidí cambiar la metrica a binary_accuracy, debido a que evaluará adecuadamente la precisión de cada atributo de forma independiente, lo que que considere apropiado para el modelo, con esto la precisión del modelo incremento, de forma significativa.
<center>
  <h4>Mejor precisión:</h4>
  <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png"/>
  </a>
</center>
<br>
<center>
  <h4>Predicción de atributos:</h4>
 <a href="https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/blob/main/atributos/seguir_entrenando.py" target="_blank">
    <img width="20%" src="https://programacion.net/files/article/20160614050620_python-logo.png"/>
  </a>
</center>

---
### Reconocimiento facial
---
Para este modelo comencé cargando imágenes de mi rostro, sin embargo, no las redimensione, lo que fue un problema porque no coincidía con las dimensiones esperadas del modelo preentrendo, además, no separe adecuadamente las imágenes en sus respectivos directorios y no los estructure de forma adecuada, lo cual me genero un error. Lo que resolví con lo siguiente:
<center> 
  <h4>Procesamiento de datos:</h4>
 <a href="https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/blob/main/Reconocimiento/redimension.py" target="_blank">
    <img width="20%" src="https://programacion.net/files/article/20160614050620_python-logo.png"/>
  </a>
</center>
<br>
Al considerar esto, cargue al modelo preentrenado ` celebA17.h5`, así como sus capas convolucionales, como se puede ver explícitamente en el código, al compilar el modelo observe que tuve un sobreajuste, por lo que, decidí aumentar los argumentos de ` ImageDataGenerator()` además de usar el regularizador L1 y L2 en el clasificador, con esto pude hacer que el modelo dejara de sobre ajustar.
<center>
  <h4>Mejor precisión:</h4>
  <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial2?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png"/>
  </a>
</center>
<br>
<center>
  <h4>Reconocimiento Facial:</h4>
 <a href="https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/blob/main/Reconocimiento/reconocimiento.py" target="_blank">
    <img width="20%" src="https://programacion.net/files/article/20160614050620_python-logo.png"/>
  </a>
</center>

## License

This project uses an [GENU License](https://github.com/Jeremy-22/Tarea-8-Reconocimento-Facial/blob/main/LICENSE).

