# -Human-Activity-Recognition-
Human Activity Recognition Using LSTM (TensorFlow) This project implements a Human Activity Recognition (HAR) system using an LSTM-based recurrent neural network built with TensorFlow. The model is trained on segmented time-series sensor data (e.g., accelerometer, gyroscope) and classifies human activities such as walking, sitting, standing, etc.

🚀 Project Overview
This notebook:
Loads and preprocesses segmented sensor data

Reshapes data into [samples, time_steps, features]

Splits the dataset into train/test sets

Builds a custom 2-layer LSTM model using TensorFlow

Trains the model using Adam optimizer

Tracks training history (loss + accuracy)

Visualizes learning curves

Generates a confusion matrix for detailed evaluation


🔧 Requirements
Install the following dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow scipy

Note:
 This project uses tf.contrib, which is available only in TensorFlow 1.x.
 If using Google Colab, enable TF1 compatibility:


📊 Data Preparation
The dataset is expected to contain:
segments → numpy array of segmented time-series input
labels → one-hot encoded activity labels
N_time_steps → number of timesteps per segment

N_features → number of sensor channels (e.g., x,y,z axes)

Example reshaping:
reshaped_segments = np.asarray(
    segments, dtype=np.float32
).reshape(-1, N_time_steps, N_features)


🧠 Model Architecture (LSTM)
The network consists of:
Dense ReLU transformation
2 stacked LSTM layers (BasicLSTMCell)
Output dense layer with softmax activation
L2 regularization and softmax cross-entropy are used for training.


⚙️ Training
Key training parameters:
Epochs: 50
Batch Size: 1024
Optimizer: Adam (lr = 0.0025)
Loss: Softmax cross-entropy + L2 loss
The notebook tracks:
Training accuracy
Test loss

🗂️ Outputs
After training, you will obtain:
Final test accuracy and loss
Loss/accuracy graphs
Confusion matrix plot
Model predictions on test data
%tensorflow1_version 1.x

🙌 Acknowledgments
This implementation is inspired by standard LSTM methods used in real-time Human Activity Recognition research and open datasets.





