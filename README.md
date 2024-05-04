# Tarea-8-Reconocimento-Facial

### Predicción de atributos de la base de datos [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (la cual contiene imágenes de rostros etiquetadas con 40 características o atributos).
---
Para realizar este modelo primero se procesaron los datos, en donde se quito el doble espacio de algunos datos, cargamos los datos y cambiamos nuestras etiquetas con valor -1 a 0, sé dividio el conjunto de datos entre validación y entrenamiento. Al realizar esto, se llamó a Wanbd.ia y se contruyo la arquitectura del modelo con capas Convolucionales 2D, con un clasificador de capas densas, y usando:
```bash
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
```
<center>
  <h4></h4>
  <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial/runs/388f5ky0?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png" alt="Open In Colab"/>
  </a>
</center>
<br>
Para obtener una mejor precisión opte por seguir entrenando la red, cambiando el batch_size, el learning_rate, aumentar capas a la arquitectura del modelo y regularizadores, pero con esto solo pude tener una precisión maxima del 34.66%.
<center>
  <h4></h4>
  <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png" alt="Open In Colab"/>
  </a>
</center>
<br>
<center>
  <h4>Predicción de atributos:</h4>
 <a href="https://wandb.ai/jeryrangmart/reconocimiento%20facial?nw=nwuserjeryrangmart" target="_blank">
    <img width="20%" src="https://import.cdn.thinkific.com/cdn-cgi/image/width=384,dpr=2,onerror=redirect/705742%2Fcustom_site_themes%2Fid%2FxVbf2a4QI6LMRQywVIhA_Group%2022406.png" alt="Open In Colab"/>
  </a>
</center>