# ImageClassificationMLP_CNN
# 🖼️ Image Classification with MLP and CNN using TensorFlow and Keras 🧠

This project implements and compares two different neural network architectures, a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), for image classification. It was originally developed as part of a Machine Learning course 📚.

## Overview 🧐

The goal of this project is to:

* Load and preprocess an image dataset (originally in MATLAB `.mat` format 💾).
* Build and train an MLP model for image classification 🤖.
* Build and train a CNN model for image classification <0xF0><0x9F><0xAA><0xB8>.
* Evaluate the performance of both models on a test dataset 📊.
* Visualize training history (loss and accuracy curves) 📈📉.
* Display sample images along with the models' predictive distributions and final predictions 👀.

## Technologies Used 💻

* **Python:** Programming language 🐍
* **TensorFlow:** Open-source machine learning framework <0xF0><0x9F><0xAB><0x9B>
* **Keras:** High-level API for building and training neural networks (integrated into TensorFlow) 🧱
* **NumPy:** Library for numerical computations <0xF0><0x9F><0x97><0x88>
* **SciPy:** Library for scientific and technical computing (used for loading `.mat` files) 🧪
* **Matplotlib:** Library for creating visualizations 📊

## Dataset 📂

The project utilizes an image dataset stored in MATLAB's `.mat` format. The dataset is split into training and testing sets.

* **`train.mat`:** Contains the training images and their corresponding labels 🚂.
* **`test.mat`:** Contains the testing images and their corresponding labels ✅.

* Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. (PDF)

**(Note: You will need to provide the actual names or descriptions of your `.mat` files if they are different.)**

## Project Structure 📂

The repository contains the following file:

* `your_notebook_name.ipynb` (or the name you give your Colab notebook 📝): This notebook contains the complete Python code for data loading, preprocessing, model building, training, evaluation, and visualization.

## Setup and Usage 🚀

1.  **Open the Colab notebook:** Open the `your_notebook_name.ipynb` file in Google Colaboratory.
2.  **Upload Dataset (if necessary):** If your `train.mat` and `test.mat` files are not already accessible (e.g., in your Google Drive), upload them to the Colab environment ☁️.
3.  **Update File Paths:** Modify the `loadmat('')` lines at the beginning of the notebook to point to the correct paths of your uploaded `.mat` files. For example:
    ```python
    train = loadmat('train.mat')  # If uploaded directly
    # Or if in Google Drive (after mounting):
    # from google.colab import drive
    # drive.mount('/content/drive')
    # train = loadmat('/content/drive/MyDrive/path/to/your/train.mat')
    ```
4.  **Run the Notebook:** Execute the cells in the Colab notebook sequentially ▶️. The code will:
    * Load and preprocess the data ⚙️.
    * Define, compile, and train the MLP model 🧠💪.
    * Define, compile, and train the CNN model <0xF0><0x9F><0xAA><0xB8>💪.
    * Display training and validation loss and accuracy curves 📈📉.
    * Evaluate the models on the test set 🧪📊.
    * Show sample test images with their true labels and the predictions from both models 👀➡️🤖<0xF0><0x9F><0xAA><0xB8>.

## Key Observations 🤔

**(After running your notebook, you can add some key observations here, such as):**

* The CNN model generally achieved higher accuracy compared to the MLP model ✅.
* The training and validation loss decreased over epochs for both models 📉.
* The learning curves provide insights into the training process and potential overfitting 📈📉🧐.
* Qualitative comparison of the predictions on sample images 👀🤖<0xF0><0x9F><0xAA><0xB8>.

## Potential Improvements ✨

* Experiment with different neural network architectures (e.g., deeper CNNs, different activation functions) 🏗️.
* Implement data augmentation techniques to improve model generalization ➕🖼️.
* Explore hyperparameter tuning to optimize model performance ⚙️🔧.
* Investigate the impact of grayscale conversion on model accuracy ⚪⚫➡️🌈.
* Use a more standardized image dataset (e.g., CIFAR-10) for easier reproducibility 🌍.
