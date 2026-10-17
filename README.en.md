# car-distance-prediction

# ADAS Prototype: Vehicle Segmentation, Lane Tracking & Distance Estimation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Mask%20RCNN-red?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## Description

This project is a prototype of an **Advanced Driver Assistance System (ADAS)** developed in Python. It combines **Deep Learning** and **Classical Computer Vision** techniques to interpret the driving environment in real time.

The main goal of the system is not only to detect vehicles, but to **compute the safety distance** specifically to the vehicles located within the driver's lane, filtering out irrelevant traffic from adjacent lanes.

## Main Features

* **Instance Segmentation (Mask R-CNN):** Accurate vehicle detection using a ResNet-50 FPN network, **fine-tuned on the Cityscapes dataset** for better accuracy in urban environments.
* **Lane Detection:** Classical computer vision algorithm (Sobel/Canny + Sliding Windows) that generates a binary mask of the current lane.
* **Distance Estimation:** Distance calculation (in meters) using the **Pinhole Camera Model**, based on the known width of each vehicle type.
* **Smart Filtering (Data Association):** Logic that intersects the segmentation masks with the lane mask to measure distance *only* to the relevant vehicles.
* **ROI Optimization (Sky Removal):** Optimization that skips processing the upper part of the image (sky/horizon) to reduce GPU load and increase FPS.

## Processing Pipeline

The system processes each video frame following this flow:

1.  **Lane Detection (Classical Vision):**
    * Image filtering (Sobel + HLS color threshold, or Canny).
    * Perspective transform ("Bird's-eye view").
    * Polynomial line fitting using Sliding Windows or prioritized search.
2.  **Vehicle Detection (Deep Learning):**
    * **Static ROI:** Cropping the upper part of the image (sky) to optimize inference.
    * Inference with **Mask R-CNN** over the region of interest.
    * Remapping of the detected coordinates back to the original frame.
3.  **Sensor Fusion:**
    * The bottom centroid of each detected vehicle is computed.
    * **Pixel-wise Check:** Checks whether that point coincides with an active pixel in the lane mask generated in step 1.
4.  **Distance Calculation:**
    * If the vehicle is in the lane, the formula $D = (W_{real} \cdot f) / W_{pixel}$ is applied.
    * A dynamic color (Red $\to$ Green) is assigned according to proximity.
5.  **Visualization:**
    * Vehicles outside the lane are marked in **Cyan**.
    * Vehicles in our trajectory display their distance and a color-coded alert.

## Installation

### Prerequisites
* Python 3.8+
* NVIDIA GPU (recommended for smooth inference with CUDA)

### Steps
1.  Clone the repository:
    ```bash
    git clone [https://github.com/tu-usuario/adas-vehicle-segmentation.git](https://github.com/tu-usuario/adas-vehicle-segmentation.git)
    cd adas-vehicle-segmentation
    ```

2.  Create a virtual environment and install the dependencies:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate  # On Windows
    # source .venv/bin/activate  # On Linux/macOS
    pip install -r requirements.txt
    pip install -e .
    ```

    `pip install -e .` installs the project's `src` package, so the scripts and notebooks can import it from any folder.

    If you don't have CUDA and don't plan to install it for GPU inference optimization, also install the following dependencies:

    ```bash
    pip install torch torchvision
    ```

    If you are going to install CUDA for your NVIDIA GPU, or already have it installed, follow the optional step below.

### (Optional) CUDA Installation for NVIDIA GPUs

    To significantly speed up model inference, it is recommended to install the **CUDA Toolkit**. Follow these steps:

1. Install the NVIDIA drivers.
2. Install the [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads).
3. Install the [PyTorch](https://pytorch.org/get-started/locally/) version compatible with your CUDA version.

#### Verify that PyTorch detects CUDA

Run the following in the terminal to check that CUDA is available:

```python
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True`.

> **Note:** If you prefer not to install CUDA, the model will run on CPU, but more slowly. Installing CUDA is entirely optional.

    

## Project Structure

```
src/                 Package with all the logic (no hard-coded paths)
  lanes/             Lane detection: filters (sobel/canny), perspective, fitting and mask
  detection/         Mask R-CNN: model construction/loading, inference and drawing
  distance/          Pinhole-model distance and vehicle-lane association
  training/          Cityscapes dataset and fine-tuning loop (vendor/ = torchvision scripts)
  pipeline.py        Combines lanes + detection + distance, for image or video
  video.py           Generic video read/write loop
scripts/             Command-line entry points
notebooks/           Step-by-step explanation of each part (01 to 05)
assets/              Example images and videos
weights/             Model weights and checkpoints (not committed to git)
data/                Cityscapes dataset (not committed to git)
outputs/             Generated results (not committed to git)
```

## Usage

The main script is `scripts/run_distances.py`, explained in `notebooks/05_distances.ipynb`. The other parts are explained in their corresponding notebooks. Run the scripts from the project root.

You must copy the files from the [Drive folder](https://drive.google.com/drive/folders/1GSkANsIEhRQM3dGJAGoPcmQ9cjh6t_2X?usp=drive_link) into your project (the model weights go in `weights/maskrcnn_cityscapes.pth`).

By default it is configured to work with a test video. You can see the result by running:
```bash
python scripts/run_distances.py --input assets/distance_prediction/videos/video1_1.mp4 --output outputs/resultado_final.mp4
```

Add `--show` to watch the video while it is being processed. Other scripts:
```bash
python scripts/run_lanes.py -i assets/road_segmentation/videos/video1_recortado.mp4 -o outputs/carril.mp4 --mode canny
python scripts/run_segmentation.py -i assets/car_segmentation/videos/video1.mp4 -o outputs/segmentacion.mp4
python scripts/train.py --data data/cityscape_data
```

If you want to try your own video, some initial setup is required:
1. Calibrate the camera that will record the videos to obtain its focal length.
2. Take a photo of a straight, flat lane, and determine the trapezoid that contains the lane from the perspective of that image (see `notebooks/01_road_lines_sobel.ipynb` to understand how to do this).

Once you have these two parameters, you can pass them as arguments to `scripts/run_distances.py` (see `--help`).
