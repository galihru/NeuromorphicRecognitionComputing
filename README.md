# Spiking Neuromorphic for Real-Time Facial Expression Recognition Application

---

## 1. Project Overview

This repository presents a **Real-Time Facial Expression Recognition System** built upon a **Spiking Neural Network (SNN)**. Inspired by the brain's computational principles, this system processes visual information from a live camera feed to identify human facial expressions (e.g., smile, neutral, surprised) in an unsupervised, adaptive manner. Utilizing Leaky Integrate-and-Fire (LIF) neurons and Spike-Timing-Dependent Plasticity (STDP), the network learns and adapts its synaptic weights based on the temporal correlation of neuronal firing, enhanced by homeostatic plasticity and lateral inhibition mechanisms to promote robust and distinct expression representations.

The project demonstrates a practical application of neuromorphic computing principles for real-world tasks, focusing on energy efficiency and biologically plausible learning.

## 2. Key Features

* **Neuromorphic Computing Paradigm:** Implements a biologically-inspired SNN architecture for real-time processing.
* **Real-Time Performance:** Processes live camera input with concurrent visualization of network dynamics (neuronal potentials, spike events, and synaptic weights).
* **Adaptive Learning:** Employs **Spike-Timing-Dependent Plasticity (STDP)** for unsupervised, online learning of facial features and their association with expressions.
* **Robust Biological Mechanisms:** Integrates **Homeostatic Plasticity (L1-norm weight normalization)** to maintain stable neuronal activity and **Lateral Inhibition** in the output layer for competitive and distinct expression classification.
* **Gabor Filter Feature Extraction:** Utilizes a bank of Gabor filters for initial feature extraction, mimicking early visual cortex processing.
* **Modular Design:** Separates the SNN simulation, real-time camera processing (OpenCV), and dynamic plotting (Matplotlib with `multiprocessing`) for enhanced performance and modularity.

## 3. System Architecture

The system comprises three main components working in parallel:

1.  **Input Preprocessing and Feature Extraction:**
    * Grayscale conversion and resizing of facial ROI.
    * Application of a diverse **Gabor filter bank** to extract orientation and scale-specific features.
    * Spatial pooling (downsampling) to create a compact feature map.

2.  **Spiking Neural Network (SNN):**
    * **Feature Neuron Layer:** A grid of LIF neurons receives input from the pooled Gabor feature map.
    * **Expression Neuron Layer:** Dedicated LIF neurons for each expression (e.g., 'smile', 'neutral', 'surprise') receive input from the feature layer.
    * **Synaptic Connections:** All-to-all excitatory connections from feature neurons to expression neurons, with weights adapted by STDP.
    * **Learning Rules:**
        * **STDP:** Updates synaptic weights based on the precise timing of pre- and post-synaptic spikes.
        * **Weight Normalization (Homeostatic Plasticity):** Ensures the sum of incoming weights to each expression neuron remains constant, promoting healthy competition and preventing runaway potentiation/depression.
        * **Lateral Inhibition:** Implemented in the expression layer, where the most active expression neuron suppresses the activity of competing neurons, leading to a "winner-take-all" like behavior.

3.  **Real-Time Visualization and User Interface:**
    * **OpenCV:** Handles camera feed acquisition, face detection, ROI extraction, and display of the processed frame with detected expression.
    * **Matplotlib (Separate Process):** Provides real-time plots of neuronal membrane potentials, spike raster plots of expression neurons, and dynamic heatmaps of synaptic weights, offering deep insights into the network's internal dynamics. This component runs in a separate process to avoid blocking the main OpenCV thread.

## 4. Mathematical Formulation

### 4.1. Leaky Integrate-and-Fire (LIF) Neuron Model

The subthreshold dynamics of the membrane potential $V_m$ for an LIF neuron are governed by the following differential equation:

$$
\tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)
$$

Where:
* $V_m$: Membrane potential
* $\tau_m$: Membrane time constant
* $V_{rest}$: Resting potential
* $R_m$: Membrane resistance
* $I(t)$: Input current at time $t$

Upon reaching a threshold $V_{th}$, the neuron fires a spike, and its membrane potential is reset to $V_{reset}$ for a refractory period $t_{ref}$.

### 4.2. Spike-Timing-Dependent Plasticity (STDP)

The change in synaptic weight $\Delta w$ between a pre-synaptic spike at $t_{pre}$ and a post-synaptic spike at $t_{post}$ is determined by the time difference $\Delta t = t_{post} - t_{pre}$:

$$
\Delta w = \begin{cases} A_{pre} \cdot e^{-\Delta t / \tau_{pre}} & \text{if } \Delta t > 0 \quad \text{(LTP - Long-Term Potentiation)} \\ A_{post} \cdot e^{\Delta t / \tau_{post}} & \text{if } \Delta t < 0 \quad \text{(LTD - Long-Term Depression)} \end{cases}
$$

Where:
* $A_{pre}$: Potentiation amplitude
* $A_{post}$: Depression amplitude (typically negative)
* $\tau_{pre}$: Potentiation time constant
* $\tau_{post}$: Depression time constant

Synaptic weights $w$ are constrained within $[w_{min}, w_{max}]$.

### 4.3. L1-Norm Weight Normalization (Homeostatic Plasticity)

To maintain stable network activity, the sum of absolute incoming weights to each post-synaptic neuron is normalized to a target sum $S_{target}$:

$$
w_{ij}^{new} = w_{ij}^{old} \cdot \frac{S_{target}}{\sum_{k} |w_{kj}^{old}|}
$$

Where:
* $w_{ij}$: Weight from pre-synaptic neuron $i$ to post-synaptic neuron $j$.
* $S_{target}$: Desired sum of absolute weights (e.g., `WEIGHT_NORMALIZATION_SUM`).

### 4.4. Lateral Inhibition

For a set of competing post-synaptic neurons (e.g., expression neurons), the input to a neuron $j$ is inhibited by the activity of the most active neuron $m$ in the group:

$$
I_{j}^{final} = I_{j}^{raw} - \alpha \cdot I_{m}^{raw} \quad \text{for } j \neq m
$$

Where:
* $I_{j}^{final}$: Final input current to neuron $j$.
* $I_{j}^{raw}$: Raw input current to neuron $j$.
* $I_{m}^{raw}$: Raw input current to the most active neuron $m$.
* $\alpha$: Lateral inhibition strength (e.g., `LATERAL_INHIBITION_STRENGTH`).

## 5. Getting Started

### 5.1. Prerequisites

* Python 3.x
* `numpy`
* `opencv-python`
* `matplotlib`

### 5.2. Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/galihru/NeuromorphicRecognitionComputing.git](https://github.com/galihru/NeuromorphicRecognitionComputing.git)
    cd Spiking-Neural-Network-Facial-Expression
    ```
2.  Install the required Python packages:
    ```bash
    pip install numpy opencv-python matplotlib
    ```

### 5.3. Running the Application

Execute the main Python script:

```bash
python your_main_script_name.py
