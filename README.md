# Code Usage Guide
This guide explains how to use the provided code to train and evaluate different models for image binary classification. The code includes implementations of three models: CNN, DenseNet, and ResNet, and provides functionality for model sparsification, quantization, and conversion to TensorFlow Lite format. The models are trained on image datasets of three different resolutions: 8x8, 16x16, and 32x32.

## Prerequisites
Before using the code, make sure you have the following:
 - Python 3.x installed
 - TensorFlow and TensorFlow Model Optimization libraries installed
 - The image datasets for training and testing prepared in separate directories (to download the Malaria dataset use: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)

## Dependencies
The code requires the following Python libraries:
 - *time*
 - *tensorflow*
 - *tensorflow.keras*
 - *tensorflow.keras.layers*
 - *tensorflow.keras.models*
 - *tensorflow_model_optimization*
 - *tempfile*
 - *numpy*

## Running the examples on Linux (Ubuntu 22.04)
To run the code from the terminal, follow these steps:
 - Open a terminal or command prompt.
 - Navigate to the directory where the code file is located using the cd command. For example, if the code file is in the "code" directory on your desktop, you would use the following command:
 ``` cd Desktop/code ```
 - Ensure that you have Python 3.x installed on your system. You can check the version by running the following command:
 ``` python --version ```
 - Install the required dependencies by running the following command:
 ``` pip install tensorflow tensorflow_model_optimization ```
 - Once the dependencies are installed, you can run the code using the following command:
 ``` python resolution_x_resolution.py ```
 - Replace "resolution_x_resolution.py" with the actual name of your code file.

The code will start executing, and you will see the output and progress in the terminal. The code will train and evaluate different models for image classification based on the provided instructions.

## Running the examples on Windows (Windows 10)
To run the code from the command prompt, follow these steps:
 - Open the Command Prompt by pressing Win + R and typing cmd, then press Enter.
 - Use the cd command to navigate to the directory where the code file is located. For example, if the code file is in the "code" directory on your desktop, you would use the following command:
 ``` cd C:\Users\YourUsername\Desktop\code ```
 - Ensure that you have Python 3.x installed on your system. You can check the version by running the following command:
 ``` python --version ```
 - Install the required dependencies by running the following command:
 ``` pip install tensorflow tensorflow_model_optimization ```
 - Once the dependencies are installed, you can run the code using the following command:
 ``` python resolution_x_resolution.py ```
 - Replace "resolution_x_resolution.py" with the actual name of your code file.

The code will start executing, and you will see the output and progress in the command prompt. The code will train and evaluate different models for image classification based on the provided instructions.
Note: Make sure you have the necessary image datasets prepared in separate directories as mentioned in the code. Also, update the code file if needed, such as adjusting file paths or dataset configurations, before running it from the terminal.
