Authors: L. Cavallaro, T. Serafin, A.Liotta;  
Proof of concept implementation of Sparse Evolutionary Training (SET) on Embdedded systems on Malaria Dataset using Keras and a mask over weights
This implementation in inspired by the work of Mocanu et al.:
@article{Mocanu2018SET,
author = {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
journal = {Nature Communications},
title = {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
year = {2018},
doi = {10.1038/s41467-018-04316-3}
}

This code and was tested with Python 3.10.0, Keras 2.12.0, Tensorflow 2.12.0, Numpy 1.23.5;
The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
For an easy understanding of the code functionality please read the following articles.
If you use parts of this code, please cite the following articles:
@article{NUTMA2023,
title={Miniaturisation of Binary Classifiers through Sparse Neural Networks},
author={Cavallaro, Lucia and Serafin, Tommaso and Liotta, Antonio},
journal={Numerical Computations: Theory and Algorithms NUMTA 2023},
pages={in press},
year={2023}
}

@mastersthesis{SerafinMSc2023,
author = {Serafin, T.},
institution = {Free University of Bozen-Bolzano - ITA},
school = {Faculty of Enginnering},
note={M.Sc. in Software Engineering for Information Systems}
title = {Towards Efficient Miniaturisation of Binary Classifiers through Sparse Neural Networks: a Trade-off Analysis},
year = {to be published},
}

# Code Usage Guide
This guide explains how to use the provided code to train and evaluate different models for image binary classification. The code includes implementations of three models: CNN, DenseNet, and ResNet, and provides functionality for model sparsification, quantization, and conversion to TensorFlow Lite format. The models are trained on image datasets of three different resolutions: 8x8, 16x16, and 32x32.

## Prerequisites
Before using the code, make sure you have the following:
 - Python 3.x installed
 - TensorFlow and TensorFlow Model Optimization libraries installed
 - The image datasets for training and testing prepared in separate directories (to download the Malaria dataset use: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)

## Dependencies
The code requires the following Python libraries:
 - time
 - tensorflow
 - tensorflow.keras
 - tensorflow.keras.layers
 - tensorflow.keras.models
 - tensorflow_model_optimization
 - tempfile
 - numpy

## Running the examples
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
Note: Make sure you have the necessary image datasets prepared in separate directories as mentioned in the code. Also, update the code file if needed, such as adjusting file paths or dataset configurations, before running it from the terminal.
