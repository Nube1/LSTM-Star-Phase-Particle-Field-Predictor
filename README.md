
# LSTM Star Phase & Particle Field Predictor

This project simulates MCNP-like (Monte Carlo N-Particle) field fluctuations across various simulated "star phases" and utilizes a Deep Learning model (Long Short-Term Memory - LSTM) to predict future field intensity based on historical sequences.

## 📌 Overview

The application generates synthetic time-series data representing particle field fluctuations modulated by star phases. It then prepares this data for a Recurrent Neural Network (RNN) to perform time-series forecasting.

**Key Features:**
*   **Synthetic Data Generation:** Simulates 12 distinct star phases with sinusoidal modulation and Gaussian noise.
*   **Data Preprocessing:** Uses MinMax Scaling and sequence generation (windowing) for LSTM ingestion.
*   **Deep Learning Model:** A stacked LSTM architecture (64 units -> 32 units) with Dropout layers to prevent overfitting.
*   **Visualization:** Plots ground truth vs. predicted values and the overall field fluctuations across phases.

## 🛠️ Dependencies

To run this project, you need Python installed along with the following libraries:

*   **NumPy:** For numerical operations and data generation.
*   **Matplotlib:** For plotting graphs.
*   **TensorFlow / Keras:** For building and training the neural network.
*   **Scikit-Learn:** For data scaling and splitting training/testing sets.

## 🚀 Installation

1.  **Clone the repository** (or download the script):
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the required packages**:
    You can install them individually via pip:
    ```bash
    pip install numpy matplotlib tensorflow scikit-learn
    ```

## 🏃 Usage

Run the script using Python:

```bash
python main.py
```
*(Replace `main.py` with whatever you named your script).*

## 🧠 How It Works

1.  **Simulation:** The code generates 1,000 samples divided into 12 phases. The signal follows a sine wave pattern multiplied by random normal noise to mimic stochastic field fluctuations.
2.  **Sequence Creation:** The data is normalized (0 to 1 range). It creates a "look-back" window of **50 steps** (sequence length). The model tries to predict the 51st step.
3.  **Model Architecture:**
    *   `LSTM (64 units, ReLU)`: Captures temporal dependencies.
    *   `Dropout (0.2)`: Regularization.
    *   `LSTM (32 units, ReLU)`: Deeper feature extraction.
    *   `Dense (1)`: Output layer for regression.
4.  **Training:** The model trains for 50 epochs using the Adam optimizer and Mean Squared Error loss.

## 📊 Results & Visualization

Upon execution, the script produces two plots:

1.  **True vs. Predicted Values:** A comparison line graph showing how well the LSTM predicted the test set data.
2.  **Field Fluctuations Across Phases:** A visualization of the entire generated dataset showing the intensity of fluctuations over the simulated phases.

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/)
```
