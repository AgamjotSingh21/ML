# 🧠 CNN vs ANN GPU & CPU Performance Benchmark

This project demonstrates a comparative study between **Convolutional
Neural Networks (CNN)** and **Artificial Neural Networks (ANN)** on
**CPU** and **GPU** using TensorFlow.

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1. Environment Setup

1.  Create and activate a virtual environment:

    ``` bash
    python -m venv tf_gpu_env
    tf_gpu_env\Scriptsctivate
    ```

2.  Install TensorFlow with GPU support (compatible with CUDA 11.2 &
    cuDNN 8.1):

    ``` bash
    pip install tensorflow==2.9.1
    ```

3.  Install essential libraries:

    ``` bash
    pip install numpy==1.24.4 pandas==2.1.1 opencv-python==4.7.0 matplotlib==3.7.1 scikit-learn==1.2.2
    ```

------------------------------------------------------------------------

## 🎥 Video Data Preparation

1.  Load your video file using OpenCV:

    ``` python
    import cv2
    cap = cv2.VideoCapture("video.mp4")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (160, 140))
        frames.append(frame_resized)
    cap.release()
    print("Frames loaded:", len(frames))
    ```

2.  Store frames as a NumPy array for model training:

    ``` python
    import numpy as np
    X = np.array(frames) / 255.0
    ```

------------------------------------------------------------------------

## 🧩 Model Architectures

### ANN Model

``` python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

ann_model = Sequential([
    Flatten(input_shape=(140, 160, 3)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### CNN Model

``` python
from tensorflow.keras.layers import Conv2D, MaxPooling2D

cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(140,160,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

------------------------------------------------------------------------

## 🧠 Running on CPU vs GPU

To force TensorFlow to use **CPU**:

``` python
with tf.device('/CPU:0'):
    ann_model.fit(X, y, epochs=5)
```

To use **GPU**:

``` python
with tf.device('/GPU:0'):
    ann_model.fit(X, y, epochs=5)
```

Repeat for both **ANN** and **CNN** models.

------------------------------------------------------------------------

## 📊 Performance Comparison

  Model   Device   Time (s)   FPS
  ------- -------- ---------- -----
  ANN     CPU      ...        ...
  ANN     GPU      ...        ...
  CNN     CPU      ...        ...
  CNN     GPU      ...        ...

### Visualization

``` python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12,6))

results.pivot(index="Model", columns="Device", values="Time (s)").plot(kind='bar', ax=ax[0])
ax[0].set_title("Execution Time (s)")
ax[0].set_ylabel("Time (s)")

results.pivot(index="Model", columns="Device", values="FPS").plot(kind='bar', ax=ax[1])
ax[1].set_title("Frames per Second (FPS)")
ax[1].set_ylabel("FPS")

plt.tight_layout()
plt.show()
```

------------------------------------------------------------------------

## 📈 Observations

-   GPU accelerates computation significantly for CNNs.
-   ANN shows smaller gains since it has fewer convolutional operations.
-   Memory usage is higher on GPU due to tensor parallelization.
-   Larger frame sizes (like 1080p) can cause memory overflow on limited
    VRAM GPUs.

------------------------------------------------------------------------

## 🧩 Conclusion

This benchmark provides insights into how **hardware acceleration**
affects **neural network performance**, helping developers optimize
model selection and deployment environments.
