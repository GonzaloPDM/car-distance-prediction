# car-distance-prediction

# ADAS Prototype: Vehicle Segmentation, Lane Tracking & Distance Estimation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Mask%20RCNN-red?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## Descripción

Este proyecto es un prototipo de **Sistema Avanzado de Asistencia al Conductor (ADAS)** desarrollado en Python. Combina técnicas de **Deep Learning** y **Visión por Computador Clásica** para interpretar el entorno de conducción en tiempo real.

El objetivo principal del sistema no es solo detectar vehículos, sino **calcular la distancia de seguridad** específicamente con los vehículos que se encuentran dentro del carril del conductor, filtrando el tráfico irrelevante de carriles adyacentes.

## Características Principales

* **Segmentación de Instancias (Mask R-CNN):** Detección precisa de vehículos utilizando una red ResNet-50 FPN, **fine-tuneada con el dataset Cityscapes** para una mejor precisión urbana.
* **Detección de Carriles:** Algoritmo de visión clásica (Sobel/Canny + Sliding Windows) que genera una máscara binaria del carril actual.
* **Estimación de Distancia:** Cálculo de la distancia (en metros) utilizando el **modelo de cámara estenopeica (Pinhole Camera Model)**, basado en el ancho conocido de cada tipo de vehículo.
* **Filtrado Inteligente (Data Association):** Lógica que cruza las máscaras de segmentación con la máscara del carril para medir la distancia *únicamente* a los vehículos relevantes.
* **Optimización ROI (Sky Removal):** Optimización que descarta el procesamiento de la zona superior de la imagen (cielo/horizonte) para reducir la carga de la GPU y aumentar los FPS.

## Pipeline de Procesamiento

El sistema procesa cada frame del video siguiendo este flujo:

1.  **Detección de Carriles (Visión Clásica):**
    * Filtrado de imagen (Sobel + Umbral de color HLS, ó Canny ).
    * Transformación de perspectiva ("Bird-eye view").
    * Ajuste polinómico de líneas mediante ventanas deslizantes (Sliding Windows) o búsqueda priorizada.
2.  **Detección de Vehículos (Deep Learning):**
    * **Static ROI:** Recorte de la zona superior de la imagen (cielo) para optimizar inferencia.
    * Inferencia con **Mask R-CNN** sobre la región de interés.
    * Re-mapeo de las coordenadas detectadas al frame original.
3.  **Fusión de Sensores:**
    * Se calcula el centroide inferior de cada vehículo detectado.
    * **Pixel-wise Check:** Se verifica si dicho punto coincide con un píxel activo en la máscara del carril generado en el paso 1.
4.  **Cálculo de Distancia:**
    * Si el vehículo está en el carril, se aplica la fórmula $D = (W_{real} \cdot f) / W_{pixel}$.
    * Se asigna un color dinámico (Rojo $\to$ Verde) según la proximidad.
5.  **Visualización:**
    * Los vehículos fuera del carril se marcan en **Cyan**.
    * Los vehículos en trayectoria muestran su distancia y alerta de color.

## Instalación

### Requisitos previos
* Python 3.8+
* GPU NVIDIA (Recomendado para inferencia fluida con CUDA)

### Pasos
1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/tu-usuario/adas-vehicle-segmentation.git](https://github.com/tu-usuario/adas-vehicle-segmentation.git)
    cd adas-vehicle-segmentation
    ```

2.  Crea un entorno virtual e instala las dependencias:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate  # En Windows
    # source .venv/bin/activate  # En Linux/macOS
    pip install -r requirements.txt
    pip install -e .
    ```

    `pip install -e .` instala el paquete `src` del proyecto, para que los scripts y los notebooks puedan importarlo desde cualquier carpeta.

    Si no tienes CUDA y no quieres instalarlo para optimizar la inferencia por GPU, instala también las siguientes dependencias:

    ```bash
    pip install torch torchvision
    ```

    Si vas a instalar CUDA para tu dispositivo con GPU NVIDIA, o ya lo tienes instalado, debes seguir el siguiente paso opcional.

### (Opcional) Instalación de CUDA para GPU NVIDIA

    Para acelerar significativamente la inferencia del modelo, es recomendable instalar **CUDA Toolkit**. Sigue estos pasos:

1. Instala los drivers de NVIDIA.
2. Instala [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads).
3. Instala la versión de [PyTorch](https://pytorch.org/get-started/locally/) compatible con tu CUDA:

#### Verificar que PyTorch detecta CUDA

Escribe en la terminal lo siguiente, para comprobar que CUDA está disponible

```python
python -c "import torch; print(torch.cuda.is_available())"
```

Debería mostrar `True`.

> **Nota:** Si prefieres no instalar CUDA, el modelo funcionará con CPU, pero será más lento. La instalación de CUDA es completamente opcional.

    

## Estructura del proyecto

```
src/                 Paquete con toda la lógica (sin rutas fijas)
  lanes/             Detección de carril: filtros (sobel/canny), perspectiva, ajuste y máscara
  detection/         Mask R-CNN: construcción/carga del modelo, inferencia y dibujo
  distance/          Distancia por modelo estenopeico y asociación vehículo-carril
  training/          Dataset Cityscapes y bucle de fine-tuning (vendor/ = scripts de torchvision)
  pipeline.py        Une carriles + detección + distancia, para imagen o vídeo
  video.py           Bucle genérico de lectura/escritura de vídeo
scripts/             Puntos de entrada por línea de comandos
notebooks/           Explicación paso a paso de cada parte (01 a 05)
assets/              Imágenes y vídeos de ejemplo
weights/             Pesos del modelo y checkpoints (no se suben a git)
data/                Dataset Cityscapes (no se sube a git)
outputs/             Resultados generados (no se suben a git)
```

## Uso

El script principal es `scripts/run_distances.py`, explicado en `notebooks/05_distances.ipynb`. El resto de partes se explican en sus notebooks correspondientes. Ejecuta los scripts desde la raíz del proyecto.

Debes copiar los archivos de la [carpeta de drive](https://drive.google.com/drive/folders/1GSkANsIEhRQM3dGJAGoPcmQ9cjh6t_2X?usp=drive_link) en tu proyecto (los pesos del modelo van en `weights/maskrcnn_cityscapes.pth`).

Por defecto está configurado para funcionar con un vídeo de prueba. Puedes ver el resultado ejecutando:
```bash
python scripts/run_distances.py --input assets/distance_prediction/videos/video1_1.mp4 --output outputs/resultado_final.mp4
```

Añade `--show` para ver el vídeo mientras se procesa. Otros scripts:
```bash
python scripts/run_lanes.py -i assets/road_segmentation/videos/video1_recortado.mp4 -o outputs/carril.mp4 --mode canny
python scripts/run_segmentation.py -i assets/car_segmentation/videos/video1.mp4 -o outputs/segmentacion.mp4
python scripts/train.py --data data/cityscape_data
```

En caso de querer probar tu propio vídeo, se requiere una configuración inicial:
1. Se debe calibrar la cámara que va a grabar los vídeos para obtener su distancia focal. 
2. Se debe tomar una fotografía de un carril recto y llano, y determinar el trapecio que contiene el carril desde la perspectiva de dicha imagen (véase `notebooks/01_road_lines_sobel.ipynb` para entender como hacerlo).

Una vez que se consiguen esos dos parámetros, se pueden pasar como argumento a `scripts/run_distances.py` (véase `--help`).
