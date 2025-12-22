# car-distance-prediction

# 🚗 ADAS Prototype: Vehicle Segmentation, Lane Tracking & Distance Estimation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Mask%20RCNN-red?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## 📖 Descripción

Este proyecto es un prototipo de **Sistema Avanzado de Asistencia al Conductor (ADAS)** desarrollado en Python. Combina técnicas de **Deep Learning** y **Visión por Computador Clásica** para interpretar el entorno de conducción en tiempo real.

El objetivo principal del sistema no es solo detectar vehículos, sino **calcular la distancia de seguridad** específicamente con los vehículos que se encuentran dentro del carril del conductor, filtrando el tráfico irrelevante de carriles adyacentes.

## ✨ Características Principales

* **🏎️ Segmentación de Instancias (Mask R-CNN):** Detección precisa de vehículos utilizando una red ResNet-50 FPN, **fine-tuneada con el dataset Cityscapes** para una mejor precisión urbana.
* **🛣️ Detección de Carriles:** Algoritmo de visión clásica (Sobel/Canny + Sliding Windows) que genera una máscara binaria del carril actual.
* **📏 Estimación de Distancia:** Cálculo de la distancia (en metros) utilizando el **modelo de cámara estenopeica (Pinhole Camera Model)**, basado en el ancho conocido de cada tipo de vehículo.
* **🎯 Filtrado Inteligente (Data Association):** Lógica que cruza las máscaras de segmentación con la máscara del carril para medir la distancia *únicamente* a los vehículos relevantes.
* **🚀 Optimización ROI (Sky Removal):** Optimización que descarta el procesamiento de la zona superior de la imagen (cielo/horizonte) para reducir la carga de la GPU y aumentar los FPS.

## 🧠 Pipeline de Procesamiento

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

## 🛠️ Instalación

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
    ```

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

    

## 🚀 Uso

El archivo principal es distances.py, que viene explicado en distances.ipynb. Los demás archivos son las dependencias que necesita distances.py para funcionar. Cada una viene explicada en su correspondiente Notebook.

Por defecto está configurado para funcionar con un vídeo de prueba. Puedes ver el resultado ejecutando:
```bash
python distances.py --input display_elements/distance_prediction/videos/video1.mp4 --output results/resultado_final.mp4
```

Debes también copiar los archivos de la [carpeta de drive](https://drive.google.com/drive/folders/1GSkANsIEhRQM3dGJAGoPcmQ9cjh6t_2X?usp=drive_link) en tu proyecto.

En caso de querer probar tu propio vídeo, se requiere una configuración inicial:
1. Se debe calibrar la cámara que va a grabar los vídeos para obtener su distancia focal. 
2. Se debe tomar una fotografía de un carril recto y llano, y determinar el trapecio que contiene el carril desde la perspectiva de dicha imagen (véase road_lines.ipynb para entender como hacerlo).

Una vez que se consiguen esos dos parámetros, se pueden pasar como argumento a distances.py (véase --help).
