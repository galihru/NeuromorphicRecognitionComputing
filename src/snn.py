import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing
from collections import deque
import os

# --- Configuration Constants ---
# Leaky Integrate-and-Fire (LIF) Neuron Parameters
LIF_DT = 0.001                      # Time step for LIF neuron updates (s)
LIF_TAU_M = 0.02                    # Membrane time constant (s)
LIF_R_M = 1.0                       # Membrane resistance (Ohms)
LIF_V_REST = -0.070                 # Resting membrane potential (Volts)
LIF_V_TH = -0.050                   # Spiking threshold voltage (Volts)
LIF_V_RESET = -0.075                # Reset voltage after a spike (Volts)
LIF_T_REF = 0.002                   # Refractory period (s)
SPIKE_BUFFER_SIZE = 1000            # Max number of past spikes to store per neuron
POTENTIAL_BUFFER_SIZE = 500         # Max number of membrane potential points to store for plotting (approx. 0.5 seconds)

# Spike-Timing-Dependent Plasticity (STDP) Parameters (requires careful tuning)
STDP_A_PRE = 0.005                  # Potentiation amplitude (adjusted smaller for gradual changes)
STDP_TAU_PRE = 0.02                 # Potentiation time constant (s)
STDP_A_POST = -0.002                # Depression amplitude (adjusted smaller for gradual changes)
STDP_TAU_POST = 0.02                # Depression time constant (s)
STDP_W_MIN = 0.001                  # Minimum synaptic weight (prevents weights from becoming exactly zero)
STDP_W_MAX = 1.0                    # Maximum synaptic weight
INITIAL_SYNAPSE_WEIGHT = 0.2        # Initial synapse weight

# Spiking Neural Network (SNN) Parameters
INPUT_IMAGE_SIZE = 48               # Size to which input face ROI is resized (pixels)
GABOR_KSIZE = 5                     # Size of the Gabor kernel (pixels)
GABOR_SIGMAS = [3.0, 5.0, 7.0]      # Sigma values for Gabor filters (controls scale/bandwidth)
GABOR_LAMBDAS = [6.0, 9.0, 12.0]    # Lambda values for Gabor filters (controls wavelength)
GABOR_GAMMA = 0.5                   # Gamma value for Gabor filters (controls spatial aspect ratio)
FEATURE_BLOCK_SIZE = 4              # Size of blocks for pooling features (pixels)
FEATURE_NEURON_INPUT_SCALE = 15.0   # Scaling factor for input to feature neurons
EXPRESSION_NEURON_INPUT_SCALE = 5.0 # Scaling factor for input to expression neurons

# --- Weight Normalization & Lateral Inhibition Parameters ---
WEIGHT_NORMALIZATION_SUM = 10.0     # Target sum for L1-norm weight normalization (homeostatic plasticity)
LATERAL_INHIBITION_STRENGTH = 0.5   # Strength of lateral inhibition (0.0 - 1.0) in the expression layer

SPIKE_COUNT_WINDOW = 0.5            # Time window (s) for counting spikes to determine dominant expression

# Camera and GUI Parameters
CAMERA_WIDTH = 640                  # Camera feed width
CAMERA_HEIGHT = 480                 # Camera feed height
FONT = cv2.FONT_HERSHEY_SIMPLEX     # Font for OpenCV text overlay
TEXT_COLOR_INFO = (0, 0, 255)       # Color for informational text (BGR: Blue)
TEXT_COLOR_EXPRESSION = (0, 255, 0) # Color for detected expression text (BGR: Green)
FACE_RECT_COLOR = (255, 0, 0)       # Color for face bounding box (BGR: Red)
FACE_RECT_THICKNESS = 2             # Thickness of face bounding box
FEATURE_VIZ_DISPLAY_SIZE = 120      # Size of the feature map visualization (pixels)

# --- 1. Neuron Model: Leaky Integrate-and-Fire (LIF) ---
class LIFNeuron:
    """
    Implements a Leaky Integrate-and-Fire (LIF) neuron model.
    The neuron integrates incoming current, its membrane potential leaks towards a resting state,
    and it fires a spike when the potential exceeds a threshold, followed by a refractory period.
    """
    def __init__(self, dt=LIF_DT, tau_m=LIF_TAU_M, R_m=LIF_R_M, V_rest=LIF_V_REST,
                 V_th=LIF_V_TH, V_reset=LIF_V_RESET, t_ref=LIF_T_REF):
        self.dt = dt            # Time step
        self.tau_m = tau_m      # Membrane time constant
        self.R_m = R_m          # Membrane resistance
        self.V_rest = V_rest    # Resting membrane potential
        self.V_th = V_th        # Spiking threshold
        self.V_reset = V_reset  # Reset potential after spike
        self.t_ref = t_ref      # Absolute refractory period
        self.ref_timer = 0.0    # Counter for refractory period
        self.V_m = V_rest       # Current membrane potential
        self.spikes = deque(maxlen=SPIKE_BUFFER_SIZE) # Stores times of recent spikes
        self.last_spike_time = -np.inf # Time of the last spike
        self.potential_history = deque(maxlen=POTENTIAL_BUFFER_SIZE) # Stores membrane potential history for plotting

    def update(self, current_input, time):
        """
        Updates the neuron's membrane potential and checks for spiking.

        Args:
            current_input (float): The total input current received by the neuron.
            time (float): The current simulation time.

        Returns:
            bool: True if the neuron fired a spike, False otherwise.
        """
        if self.ref_timer > 0:
            # If within refractory period, do not integrate input
            self.ref_timer -= self.dt
            self.potential_history.append(self.V_m)
            return False

        # Integrate membrane potential using the LIF equation
        dV_m_dt = (-(self.V_m - self.V_rest) + self.R_m * current_input) / self.tau_m
        self.V_m += dV_m_dt * self.dt
        self.potential_history.append(self.V_m)

        # Check for spike condition
        if self.V_m >= self.V_th:
            self.V_m = self.V_reset        # Reset potential
            self.ref_timer = self.t_ref    # Start refractory period
            self.spikes.append(time)       # Record spike time
            self.last_spike_time = time    # Update last spike time
            return True
        return False

# --- 2. Synapse Model ---
class Synapse:
    """
    Represents a single synaptic connection with a modifiable weight.
    """
    def __init__(self, weight):
        self.weight = weight # Synaptic weight, determining strength of connection

# --- 3. Learning Rule: Spike-Timing-Dependent Plasticity (STDP) ---
class STDP:
    """
    Implements the Spike-Timing-Dependent Plasticity (STDP) learning rule.
    Synaptic weights are adjusted based on the relative timing of pre-synaptic
    and post-synaptic spikes.
    """
    def __init__(self, A_pre=STDP_A_PRE, tau_pre=STDP_TAU_PRE, A_post=STDP_A_POST,
                 tau_post=STDP_TAU_POST, w_min=STDP_W_MIN, w_max=STDP_W_MAX):
        self.A_pre = A_pre          # Amplitude for potentiation (LTP)
        self.tau_pre = tau_pre      # Time constant for potentiation
        self.A_post = A_post        # Amplitude for depression (LTD)
        self.tau_post = tau_post    # Time constant for depression
        self.w_min = w_min          # Minimum allowed synaptic weight
        self.w_max = w_max          # Maximum allowed synaptic weight

    def apply_stdp(self, synapse, t_pre, t_post):
        """
        Applies the STDP rule to a given synapse based on pre- and post-synaptic spike times.

        Args:
            synapse (Synapse): The synapse object to be updated.
            t_pre (float): Time of the last pre-synaptic spike.
            t_post (float): Time of the last post-synaptic spike.
        """
        if t_pre == -np.inf or t_post == -np.inf:
            # Skip STDP if either neuron hasn't fired yet
            return

        delta_t = t_post - t_pre # Time difference between post- and pre-synaptic spikes

        delta_w = 0
        if delta_t > 0:
            # Post-synaptic spike occurs after pre-synaptic: Potentiation (LTP)
            delta_w = self.A_pre * np.exp(-delta_t / self.tau_pre)
        elif delta_t < 0:
            # Pre-synaptic spike occurs after post-synaptic: Depression (LTD)
            delta_w = self.A_post * np.exp(delta_t / self.tau_post)

        # Update and clip the synapse weight within defined bounds
        synapse.weight = np.clip(synapse.weight + delta_w, self.w_min, self.w_max)

# --- 4. Spiking Neural Network for Face Expression Recognition ---
class FaceExpressionSNN:
    """
    Manages the architecture and dynamics of the Spiking Neural Network
    for real-time facial expression recognition. It includes feature extraction,
    neuron layers, synaptic connections, and learning rules.
    """
    def __init__(self, input_size=INPUT_IMAGE_SIZE, dt=LIF_DT):
        self.dt = dt
        self.input_size = input_size

        self.orientation_bins = 8 # Number of Gabor filter orientations
        self.gabor_filters = self._create_gabor_bank() # Bank of Gabor filters for feature extraction

        # Feature neuron layer dimensions
        self.feature_neuron_rows = input_size // FEATURE_BLOCK_SIZE
        self.feature_neuron_cols = input_size // FEATURE_BLOCK_SIZE
        # Initialize 2D array of LIF neurons for feature representation
        self.feature_neurons = [
            [LIFNeuron(dt=dt) for _ in range(self.feature_neuron_cols)]
            for _ in range(self.feature_neuron_rows)
        ]

        # Expression neuron layer: Maps feature activations to specific expressions
        self.expression_neurons = {
            'smile': LIFNeuron(dt=dt),
            'neutral': LIFNeuron(dt=dt),
            'surprised': LIFNeuron(dt=dt),
            # Extend with more expressions (e.g., 'sad', 'angry') as needed.
            # Note: Adding more expressions increases the complexity for Gabor filters and STDP to differentiate them.
        }
        self.expression_names = list(self.expression_neurons.keys()) # List of supported expression names

        # Synaptic connections from feature neurons to expression neurons
        # Each expression neuron receives input from ALL feature neurons
        self.synapses = {
            expr: [
                [Synapse(np.random.rand() * INITIAL_SYNAPSE_WEIGHT) # Initialize weights randomly
                 for _ in range(self.feature_neuron_cols)]
                for _ in range(self.feature_neuron_rows)
            ] for expr in self.expression_neurons
        }

        self.stdp_rule = STDP() # STDP learning rule instance
        # Stores the last spike time for each expression neuron, used in STDP
        self.last_expression_spike = {expr: -np.inf for expr in self.expression_neurons}

    def _create_gabor_bank(self):
        """
        Generates a bank of Gabor filters with varying orientations, sigmas, and lambdas.
        These filters are used to extract oriented edge and texture features from face images.
        """
        filters = []
        for theta in np.linspace(0, np.pi, self.orientation_bins, endpoint=False):
            for sigma in GABOR_SIGMAS:
                for lambd in GABOR_LAMBDAS:
                    kernel = cv2.getGaborKernel(
                        (GABOR_KSIZE, GABOR_KSIZE),
                        sigma,
                        theta,
                        lambd,
                        GABOR_GAMMA,
                        0, # phase offset
                        ktype=cv2.CV_32F
                    )
                    filters.append(kernel)
        print(f"Generated {len(filters)} Gabor filters.")
        return filters

    def _preprocess_frame(self, frame):
        """
        Preprocesses a raw frame (face ROI) for SNN input.
        Converts to grayscale, resizes, and normalizes pixel values.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.input_size, self.input_size))
        return resized / 255.0 # Normalize pixel values to [0, 1]

    def _extract_features(self, preprocessed_frame):
        """
        Extracts features from the preprocessed frame using the Gabor filter bank
        and spatial pooling.
        """
        frame_float32 = np.float32(preprocessed_frame)

        all_gabor_responses = []
        for kernel in self.gabor_filters:
            # Apply each Gabor filter and take the absolute response
            filtered = cv2.filter2D(frame_float32, cv2.CV_32F, kernel)
            all_gabor_responses.append(np.abs(filtered))

        # Combine responses from all Gabor filters for each pixel
        # Using mean to reduce the dimensionality of the filter responses at each pixel
        combined_response = np.stack(all_gabor_responses, axis=-1).mean(axis=-1)

        feature_map = np.zeros((self.feature_neuron_rows, self.feature_neuron_cols))
        # Perform spatial pooling to reduce the feature map size to match feature neuron grid
        for i in range(self.feature_neuron_rows):
            for j in range(self.feature_neuron_cols):
                block_start_row = i * FEATURE_BLOCK_SIZE
                block_end_row = (i + 1) * FEATURE_BLOCK_SIZE
                block_start_col = j * FEATURE_BLOCK_SIZE
                block_end_col = (j + 1) * FEATURE_BLOCK_SIZE

                # Calculate the mean response within each block
                block_mean = np.mean(combined_response[block_start_row:block_end_row,
                                                       block_start_col:block_end_col])
                feature_map[i, j] = block_mean

        return feature_map

    def normalize_weights_l1(self, expression_name):
        """
        Normalizes the incoming synaptic weights to a specific expression neuron
        using L1-norm normalization (sum of absolute weights is constrained).
        This implements a form of homeostatic plasticity, preventing runaway
        potentiation or depression of weights and promoting stable learning.
        """
        weights_for_expr = np.array([
            [s.weight for s in row] for row in self.synapses[expression_name]
        ])

        # Calculate the current sum of absolute weights
        current_sum = np.sum(np.abs(weights_for_expr))

        if current_sum > 0:
            scale_factor = WEIGHT_NORMALIZATION_SUM / current_sum
            # Apply scaling to each weight, ensuring it stays within min/max bounds
            for i in range(self.feature_neuron_rows):
                for j in range(self.feature_neuron_cols):
                    original_weight = self.synapses[expression_name][i][j].weight
                    new_weight = original_weight * scale_factor
                    self.synapses[expression_name][i][j].weight = np.clip(new_weight, STDP_W_MIN, STDP_W_MAX)

    def update(self, frame, current_time):
        """
        Performs a single simulation step for the SNN.
        Includes feature extraction, updating feature neurons,
        calculating and applying lateral inhibition, updating expression neurons,
        and applying STDP and weight normalization.

        Args:
            frame (np.ndarray): The current video frame (face ROI).
            current_time (float): The current simulation time.

        Returns:
            tuple: A dictionary indicating which expression neurons fired (boolean)
                   and the extracted feature map (np.ndarray).
        """
        preprocessed = self._preprocess_frame(frame)
        feature_map = self._extract_features(preprocessed)

        input_spikes = np.zeros((self.feature_neuron_rows, self.feature_neuron_cols), dtype=bool)

        # Update feature neurons and record their spikes
        for i in range(self.feature_neuron_rows):
            for j in range(self.feature_neuron_cols):
                current_input = feature_map[i, j] * FEATURE_NEURON_INPUT_SCALE
                if self.feature_neurons[i][j].update(current_input, current_time):
                    input_spikes[i, j] = True

        expression_output = {expr: False for expr in self.expression_neurons}

        # Calculate potential input for all expression neurons BEFORE their update.
        # This is crucial for correctly applying lateral inhibition, as inhibition
        # needs to be based on the potential strength of all competitors.
        potential_inputs = {}
        for expr, neuron in self.expression_neurons.items():
            input_sum = 0.0
            for i in range(self.feature_neuron_rows):
                for j in range(self.feature_neuron_cols):
                    if input_spikes[i, j]:
                        input_sum += self.synapses[expr][i][j].weight * EXPRESSION_NEURON_INPUT_SCALE
            potential_inputs[expr] = input_sum

        # Identify the expression neuron with the highest potential input.
        # This neuron is considered the "winner" and will inhibit others.
        if potential_inputs:
            max_input_expr = max(potential_inputs, key=potential_inputs.get)

            # --- Apply Lateral Inhibition ---
            # Calculate inhibitory currents: The most active neuron inhibits all others.
            inhibitory_currents = {}
            for expr_inhib, _ in self.expression_neurons.items():
                if expr_inhib != max_input_expr:
                    # Other neurons receive an inhibitory current proportional to the winner's input
                    inhibitory_currents[expr_inhib] = potential_inputs[max_input_expr] * LATERAL_INHIBITION_STRENGTH
                else:
                    inhibitory_currents[expr_inhib] = 0 # The dominant neuron does not inhibit itself

        # Update expression neurons, incorporating lateral inhibition
        for expr, neuron in self.expression_neurons.items():
            final_input = potential_inputs[expr] - inhibitory_currents.get(expr, 0)

            # Ensure input does not become negative after inhibition, unless
            # your specific LIF model allows for negative input currents.
            final_input = max(0, final_input)

            if neuron.update(final_input, current_time):
                expression_output[expr] = True
                self.last_expression_spike[expr] = current_time

                # Apply STDP only to the synapses connected to the neuron that just fired.
                for i in range(self.feature_neuron_rows):
                    for j in range(self.feature_neuron_cols):
                        if input_spikes[i, j]: # If the pre-synaptic neuron also fired
                            t_pre = self.feature_neurons[i][j].last_spike_time
                            self.stdp_rule.apply_stdp(
                                self.synapses[expr][i][j],
                                t_pre,
                                current_time # t_post is the current time when the expression neuron fired
                            )
                # Apply weight normalization immediately after STDP for stability
                self.normalize_weights_l1(expr)

        return expression_output, feature_map

# --- Matplotlib Visualizer Class (runs in a separate process for performance) ---
class SNNPlotter:
    """
    Manages the real-time plotting of SNN dynamics using Matplotlib.
    It runs in a separate process to avoid blocking the main OpenCV video stream.
    Displays membrane potentials, spike events, and synaptic weight heatmaps.
    """
    def __init__(self, data_queue, dt_snn, expression_neuron_names, feature_rows, feature_cols):
        self.data_queue = data_queue
        self.dt_snn = dt_snn
        self.expression_neuron_names = expression_neuron_names
        self.feature_rows = feature_rows
        self.feature_cols = feature_cols

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 9))
        self.lines = {} # For membrane potential plots
        self.spike_scatters = {} # For spike raster plots
        self.imshow_weights = {} # For synaptic weight heatmaps

        self._setup_plots()

    def _setup_plots(self):
        """Initializes the Matplotlib plots with titles, labels, and empty data."""
        # Plot 1: Expression Neuron Membrane Potentials
        self.axs[0, 0].set_title("Expression Neuron Membrane Potentials")
        self.axs[0, 0].set_xlabel("Relative Time (s)")
        self.axs[0, 0].set_ylabel("Potential (V)")
        self.axs[0, 0].set_ylim(LIF_V_RESET - 0.005, LIF_V_TH + 0.005)
        for expr in self.expression_neuron_names:
            line, = self.axs[0, 0].plot([], [], label=expr)
            self.lines[f'V_m_{expr}'] = line
        self.axs[0, 0].legend()
        self.axs[0, 0].grid(True)

        # Plot 2: Expression Neuron Spikes (Simple Raster Plot)
        self.axs[0, 1].set_title("Expression Neuron Spikes")
        self.axs[0, 1].set_xlabel("Time (s)")
        self.axs[0, 1].set_ylabel("Neuron")
        self.axs[0, 1].set_yticks(range(len(self.expression_neuron_names)))
        self.axs[0, 1].set_yticklabels(self.expression_neuron_names)
        for i, expr in enumerate(self.expression_neuron_names):
            scatter = self.axs[0, 1].scatter([], [], marker='|', s=200, label=expr)
            self.spike_scatters[f'spikes_{expr}'] = scatter
        self.axs[0, 1].set_xlim(0, SPIKE_COUNT_WINDOW * 2) # Initial time range for visibility

        # Plot 3: Synaptic Weights (Feature -> Smile) Heatmap
        self.axs[1, 0].set_title("Synaptic Weights (Features -> Smile)")
        initial_weights_smile = np.zeros((self.feature_rows, self.feature_cols)) # Initialize with zeros
        im_smile = self.axs[1, 0].imshow(initial_weights_smile, cmap='hot', vmin=STDP_W_MIN, vmax=STDP_W_MAX, interpolation='nearest')
        self.fig.colorbar(im_smile, ax=self.axs[1, 0])
        self.imshow_weights['smile'] = im_smile

        # Plot 4: Synaptic Weights (Feature -> Neutral) Heatmap
        self.axs[1, 1].set_title("Synaptic Weights (Features -> Neutral)")
        initial_weights_neutral = np.zeros((self.feature_rows, self.feature_cols)) # Initialize with zeros
        im_neutral = self.axs[1, 1].imshow(initial_weights_neutral, cmap='hot', vmin=STDP_W_MIN, vmax=STDP_W_MAX, interpolation='nearest')
        self.fig.colorbar(im_neutral, ax=self.axs[1, 1])
        self.imshow_weights['neutral'] = im_neutral

        plt.tight_layout() # Adjust plot parameters for a tight layout

    def update_plot(self, frame):
        """
        Callback function for Matplotlib animation. Fetches data from the queue
        and updates all plots.
        """
        while not self.data_queue.empty():
            try:
                plot_data = self.data_queue.get_nowait()
                current_time = plot_data['current_time']
                potential_data = plot_data['potential_data']
                spike_data = plot_data['spike_data']
                weights_data = plot_data['weights_data']

                # Update Membrane Potential plots
                for expr, history in potential_data.items():
                    if history:
                        x_data = np.arange(len(history)) * self.dt_snn
                        self.lines[f'V_m_{expr}'].set_data(x_data, history)
                        self.axs[0, 0].set_xlim(x_data[0], x_data[-1])

                # Update Spike Raster plots
                for i, expr in enumerate(self.expression_neuron_names):
                    spike_times = np.array(spike_data[expr])
                    # Only show spikes within the visible time window
                    relevant_spikes = spike_times[spike_times > current_time - (POTENTIAL_BUFFER_SIZE * self.dt_snn)]
                    y_data = np.full_like(relevant_spikes, i)
                    self.spike_scatters[f'spikes_{expr}'].set_offsets(np.c_[relevant_spikes, y_data])
                    self.axs[0, 1].set_xlim(current_time - (POTENTIAL_BUFFER_SIZE * self.dt_snn), current_time)

                # Update Synaptic Weights heatmaps (ensure keys exist before updating)
                if 'smile' in weights_data:
                    self.imshow_weights['smile'].set_array(weights_data['smile'])
                if 'neutral' in weights_data:
                    self.imshow_weights['neutral'].set_array(weights_data['neutral'])
                # If you add 'surprised' to the visualization, add it here too:
                # if 'surprised' in weights_data:
                #     self.imshow_weights['surprised'].set_array(weights_data['surprised'])

            except Exception as e:
                print(f"Error updating plot from queue: {e}")
                break

        return [] # Return empty list as per FuncAnimation requirement

# Function to run the Matplotlib plotter in a separate process
def run_plotter(data_queue, dt_snn, expression_neuron_names, feature_rows, feature_cols):
    """
    Entry point for the separate process that runs the Matplotlib visualization.
    """
    plotter = SNNPlotter(data_queue, dt_snn, expression_neuron_names, feature_rows, feature_cols)
    # FuncAnimation continuously calls update_plot to animate the figures
    ani = animation.FuncAnimation(plotter.fig, plotter.update_plot, interval=50, blit=False, cache_frame_data=False)
    plt.show()

# --- 5. Real-time Processing with Camera ---
def main():
    """
    Main function to initialize the SNN, camera, and start real-time processing
    and visualization.
    """
    use_xvfb = os.environ.get('USE_XVFB', 'false').lower() == 'true'
    
    if use_xvfb:
        os.environ['DISPLAY'] = ':99'
    snn = FaceExpressionSNN() # Initialize the SNN

    # Create a multiprocessing Queue to pass data from the main process to the plotter process
    plot_data_queue = multiprocessing.Queue()

    # Get SNN configuration details needed by the plotter
    expr_names = list(snn.expression_neurons.keys())
    feature_rows = snn.feature_neuron_rows
    feature_cols = snn.feature_neuron_cols

    # Start the Matplotlib plotter in a separate process
    plotter_process = multiprocessing.Process(
        target=run_plotter,
        args=(plot_data_queue, snn.dt, expr_names, feature_rows, feature_cols)
    )
    plotter_process.start()
    print("[INFO] Matplotlib plotting process started.")

    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Failed to open camera. Ensure camera is connected and not in use by other applications.")
        plotter_process.terminate()
        plotter_process.join() # Wait for the plotter process to finish
        return

    main.start_time = time.time() # Record start time for simulation timekeeping
    font = FONT # Set font for OpenCV text

    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("[ERROR] 'haarcascade_frontalface_default.xml' not found. Ensure OpenCV is correctly installed.")
        plotter_process.terminate()
        plotter_process.join()
        return

    print("[INFO] Application started. Press 'q' to quit.")

    last_plot_update_time = time.time()
    plot_update_interval = 0.05 # How often data is sent to the plotter (in seconds)

    # Main loop for real-time video processing
    while True:
        ret, frame = cap.read() # Read a frame from the camera
        if not ret:
            print("[ERROR] Failed to read frame from camera. Terminating application.")
            break

        frame = cv2.flip(frame, 1) # Flip frame horizontally (mirror effect)
        current_time = time.time() - main.start_time # Calculate current simulation time
        display_frame = frame.copy() # Create a copy for displaying overlays
        expression_status = "No Face Detected" # Default status text

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Detect faces

        if len(faces) > 0:
            (x, y, w, h) = faces[0] # Get coordinates of the first detected face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), FACE_RECT_COLOR, FACE_RECT_THICKNESS) # Draw bounding box

            face_roi = frame[y:y+h, x:x+w] # Extract Region of Interest (ROI) for the face

            # Update SNN with the face ROI and get expression outputs and feature map
            expressions_fired, feature_map = snn.update(face_roi, current_time)

            # Visualize the extracted feature map
            feature_viz = cv2.resize(feature_map, (FEATURE_VIZ_DISPLAY_SIZE, FEATURE_VIZ_DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
            feature_viz = cv2.normalize(feature_viz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Normalize to 0-255
            feature_viz = cv2.applyColorMap(feature_viz, cv2.COLORMAP_JET) # Apply a colormap

            # Place the feature map visualization on the display frame
            if display_frame.shape[0] > FEATURE_VIZ_DISPLAY_SIZE + 10 and display_frame.shape[1] > FEATURE_VIZ_DISPLAY_SIZE + 10:
                display_frame[10:10 + FEATURE_VIZ_DISPLAY_SIZE, 10:10 + FEATURE_VIZ_DISPLAY_SIZE] = feature_viz

            dominant_expr_name = "Neutral" # Default dominant expression
            # Calculate spike counts for all expression neurons within the recent window
            current_spike_counts = {}
            for expr, neuron in snn.expression_neurons.items():
                current_spike_counts[expr] = sum(1 for t in neuron.spikes if current_time - t < SPIKE_COUNT_WINDOW)

            # Determine the dominant expression based on which neuron spiked most
            if any(count > 0 for count in current_spike_counts.values()):
                dominant_expr_name = max(current_spike_counts, key=current_spike_counts.get)
            else:
                dominant_expr_name = "Undetected" # If no significant spiking activity

            # Display the detected expression on the frame
            cv2.putText(display_frame, f"Expression: {dominant_expr_name.capitalize()}", (x, y-10), font, 0.7, TEXT_COLOR_EXPRESSION, 2)
            expression_status = dominant_expr_name.capitalize() # Update status for general info text

            # --- Send Data to Plotter Process via Queue ---
            # Data is sent at a controlled interval to avoid overwhelming the queue
            if time.time() - last_plot_update_time >= plot_update_interval:
                potential_data_for_plot = {expr: list(neuron.potential_history)
                                            for expr, neuron in snn.expression_neurons.items()}
                spike_data_for_plot = {expr: list(neuron.spikes)
                                        for expr, neuron in snn.expression_neurons.items()}
                # Prepare weight data for plotting (only for 'smile' and 'neutral' as shown in plotter)
                weights_data_for_plot = {
                    'smile': np.array([[s.weight for s in row] for row in snn.synapses['smile']]),
                    'neutral': np.array([[s.weight for s in row] for row in snn.synapses['neutral']])
                }
                # Add 'surprised' weights if you enable its plot in SNNPlotter
                # 'surprised': np.array([[s.weight for s in row] for row in snn.synapses['surprised']])

                try:
                    # Attempt to put data into the queue without blocking
                    plot_data_queue.put_nowait({
                        'current_time': current_time,
                        'potential_data': potential_data_for_plot,
                        'spike_data': spike_data_for_plot,
                        'weights_data': weights_data_for_plot
                    })
                except multiprocessing.queues.Full:
                    # If queue is full, skip sending data for this frame to avoid slowdown
                    pass

                last_plot_update_time = time.time() # Reset plot update timer


        # Display overall application status (e.g., if no face is detected)
        cv2.putText(display_frame, f"Status: {expression_status}", (10, 10 + FEATURE_VIZ_DISPLAY_SIZE + 20), font, 0.6, TEXT_COLOR_INFO, 2)

        # Show the processed video frame
        cv2.imshow('Face Expression SNN', display_frame)

        # Check for 'q' key press to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Application terminated by user.")
            break

    # --- Cleanup ---
    cap.release() # Release camera resources
    cv2.destroyAllWindows() # Close all OpenCV windows

    plotter_process.terminate() # Terminate the Matplotlib plotting process
    plotter_process.join()      # Wait for the plotter process to finish
    print("[INFO] Matplotlib plotting process terminated.")
    print("[INFO] Application finished.")

if __name__ == "__main__":
    # Required for multiprocessing to work correctly on Windows when creating new processes
    multiprocessing.freeze_support()
    main()
