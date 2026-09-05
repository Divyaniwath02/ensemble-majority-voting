# Ensemble Majority Voting for Image Classification

## 📌 Project Overview

This project demonstrates a simple **Ensemble Learning Model using Majority Voting** for handwritten digit classification.

Three different Deep Learning models are trained on the **MNIST handwritten digit dataset**:

* Simple Neural Network
* Deep Neural Network
* Convolutional Neural Network (CNN)

The predictions from all three models are combined using **Majority Voting** to generate the final prediction.

## 🎯 Objective

To build a simple ensemble model using multiple Neural Network and Deep Learning models and combine their predictions using Majority Voting.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* MNIST Dataset

## 🔄 Project Workflow

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
 ┌──────────┬──────────┬──────────┐
 ↓          ↓          ↓
Simple NN  Deep NN     CNN
 ↓          ↓          ↓
Prediction Prediction Prediction
 └──────────┬──────────┘
            ↓
     Majority Voting
            ↓
    Final Prediction
```

## 🧠 Models Used

### 1. Simple Neural Network

A basic neural network with one hidden layer containing 128 neurons.

### 2. Deep Neural Network

A deeper neural network containing multiple hidden layers with 256, 128 and 64 neurons.

### 3. Convolutional Neural Network

A CNN containing convolution and pooling layers, designed for image classification.

## 🗳️ Majority Voting

Each model predicts a digit from 0 to 9.

For example:

```text
Simple NN → 7
Deep NN   → 7
CNN       → 1

Final Prediction → 7
```

Since two models predicted `7`, the majority voting system selects `7` as the final prediction.

## 📊 Results

| Model                 | Accuracy |
| --------------------- | -------: |
| Simple Neural Network |   96.51% |
| Deep Neural Network   |   97.23% |
| CNN                   |   98.51% |
| Ensemble Voting       |   98.01% |

The CNN achieved the highest individual accuracy, while the ensemble successfully combined predictions from all three models using majority voting.

## 📈 Visualizations

The project generates:

* `accuracy_comparison.png` – Accuracy comparison of all models
* `confusion_matrix.png` – Confusion matrix of the ensemble model
* `sample_predictions.png` – Sample MNIST predictions

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Divyaniwath02/ensemble-majority-voting.git
```

### 2. Open the project

```bash
cd ensemble-majority-voting
```

### 3. Create and activate virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 5. Run the project

```bash
python ensemble_majority_voting.py
```

## 💡 Applications

Ensemble learning and majority voting can be used in:

* Image Classification
* Pattern Recognition
* Handwritten Digit Recognition
* Medical Image Analysis
* Object Classification
* Machine Learning Decision Systems

## 📌 Conclusion

This project demonstrates how multiple Neural Network and Deep Learning models can be combined using **Majority Voting** to create an ensemble system. The approach provides a simple way to combine predictions from different models and demonstrates the practical concept of ensemble learning.
