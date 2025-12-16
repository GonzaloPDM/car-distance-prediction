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

* **🏎️ Segmentación de Instancias (Mask R-CNN):** Detección precisa de siluetas de vehículos utilizando una red neuronal convolucional (ResNet-50 FPN) pre-entrenada.
* **🛣️ Detección de Carriles:** Algoritmo de visión clásica (procesamiento de color + detección de bordes) para identificar los límites del carril actual y generar una "zona de conducción".
* **📏 Estimación de Distancia:** Cálculo de la distancia (en metros) hacia los vehículos detectados basándose en la geometría proyectiva y la posición del vehículo en el plano de la imagen.
* **🎯 Filtrado Inteligente:** Lógica de asociación de datos para medir la distancia *únicamente* a los coches que interfieren en la trayectoria (dentro del polígono del carril).
* **🚀 Optimización ROI (Smart Tracking):** Implementación de "Region of Interest" dinámica. Tras la detección inicial, el modelo restringe la búsqueda a áreas específicas para aumentar los FPS y reducir la carga de la GPU.

## 📷 Demo / Resultados

*(Sustituye esta línea con un GIF o imagen de tu proyecto funcionando)*
![Demo del Proyecto](assets/demo_result.gif)

## 🧠 Pipeline de Procesamiento

El sistema procesa cada frame del video siguiendo este flujo:

1.  **Detección de Carriles (CPU):**
    * Pre-procesamiento (Escala de grises, ROI trapezoidal).
    * Filtrado de color (Blanco/Amarillo).
    * Detección de líneas y cálculo del polígono del carril.
2.  **Detección de Vehículos (GPU):**
    * Inferencia con Mask R-CNN.
    * Si hay detecciones previas, se aplica **ROI Tracking** para buscar solo en zonas probables.
3.  **Fusión de Sensores (Lógica):**
    * Se calcula el punto de contacto de cada vehículo con el suelo (bounding box `y_max`).
    * Se verifica geométricamente (`pointPolygonTest`) si el vehículo está dentro del carril detectado.
4.  **Cálculo de Distancia:**
    * Se aplica una transformación de perspectiva (basada en la altura de la cámara y el horizonte) para convertir píxeles a metros.
5.  **Visualización:**
    * Renderizado de máscaras, cajas, carril y etiquetas de distancia sobre el frame original.

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

    Si vas a instalar CUDA para tu dispositivo, o ya lo tienes instalado y quieres utilizarlo, salta al siguiente paso opcional.
    Si no, instala también las siguientes dependencias:

    ```bash
    pip install torch torchvision
    ```

### (Opcional) Instalación de CUDA para GPU NVIDIA

    Para acelerar significativamente la inferencia del modelo, es recomendable instalar **CUDA Toolkit**. Sigue estos pasos:

#### Paso 1: Verificar GPU NVIDIA

Abre PowerShell y ejecuta:
```bash
nvidia-smi
```

Si aparece la información de tu GPU, ya tienes los drivers instalados. Si no, descárgalos desde [NVIDIA Drivers](https://www.nvidia.com/Download/driverDetails.aspx).

Comprueba la versión de CUDA de tu GPU, y descarga torch y torchvision desde la web oficial de [PyTorch](https://pytorch.org/get-started/locally/)

Ejecutarás un comando similar a:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### Paso 2: Descargar CUDA Toolkit

1. Ve a [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Selecciona:
- **Operating System:** Windows
- **Architecture:** x86_64
- **Version:** La versión que utilizaste en el paso anterior
- **Installer Type:** exe (local)
3. Descarga el archivo (aproximadamente 2.5 GB)

#### Paso 3: Instalar CUDA Toolkit

1. Ejecuta el instalador descargado
2. Acepta los términos de licencia
3. Selecciona **Custom** para la instalación
4. Asegúrate de instalar:
- ✅ CUDA Toolkit
- ✅ cuDNN (si está disponible)
- ✅ NVIDIA Nsight Compute (opcional)
5. Usa las ubicaciones de instalación por defecto (usualmente `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`)
6. Completa la instalación

#### Paso 4: Verificar instalación de CUDA

En PowerShell, ejecuta:
```bash
nvcc --version
```

Deberías ver algo como: `nvcc: NVIDIA (R) Cuda compiler driver, Version 13.0`

#### Paso 5: Instalar cuDNN (Opcional pero Recomendado)

1. Descarga cuDNN desde [NVIDIA cuDNN](https://developer.nvidia.com/rdnn) (requiere cuenta NVIDIA)
2. Extrae el contenido
3. Copia los archivos a la carpeta de CUDA:
- De `cuDNN\bin\*` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- De `cuDNN\lib\x64\*` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64`
- De `cuDNN\include\*` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include`

#### Paso 6: Actualizar PyTorch para CUDA (En tu venv activado)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Paso 7: Verificar que PyTorch detecta CUDA

```python
python -c "import torch; print(torch.cuda.is_available())"
```

Debería mostrar `True`.

> **Nota:** Si prefieres no instalar CUDA, el modelo funcionará con CPU, pero será más lento. La instalación de CUDA es completamente opcional.

    

## 🚀 Uso

Para ejecutar el procesador de video principal:

```bash
python main.py --input data/video_entrada.mp4 --output results/resultado_final.mp4
