# Alzheimer-s-MRI-Classification-with-CNN-Self-Attention

This project implements a deep learning model to classify brain MRI scans into four stages of Alzheimer's disease. It uses a **Convolutional Neural Network (CNN)**, specifically MobileNetV2, as a base and enhances it with a **Scaled Dot-Product Attention** mechanism to improve performance.

## Table of Contents

1. [Project Goal](#project-goal)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
    - [Step 1: Data Loading \& Preprocessing](#step-1-data-loading--preprocessing)
    - [Step 2: Model Architecture](#step-2-model-architecture)
    - [Step 3: Training](#step-3-training)
    - [Step 4: Evaluation](#step-4-evaluation)
4. [How to Run](#how-to-run)
5. [Results](#results)
6. [Conclusion \& Future Work](#conclusion--future-work)

## Project Goal

The main objective is to build an accurate and reliable classifier to distinguish between four stages of Alzheimer's from MRI scans:

1. **Non-Demented**
2. **Very Mild Demented**
3. **Mild Demented**
4. **Moderate Demented**

The project focuses on leveraging transfer learning and attention mechanisms for a medical imaging task.

## Dataset

The model is trained on the **Augmented Alzheimer MRI Dataset** from Kaggle, accessed via `kagglehub`.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
- **Structure:**
    - **Training/Validation:** The augmented dataset (`AugmentedAlzheimerDataset`) is split 80/20 for training and validation.
    - **Testing:** The original, un-augmented images (`OriginalDataset`) are used as the final test set to get a realistic measure of the model's performance.


## Methodology

The project follows a standard deep learning pipeline.

### Step 1: Data Loading \& Preprocessing

- **Image Loading:** Images are loaded from their respective directories using TensorFlow's `ImageDataGenerator`.
- **Image Resizing:** All images are resized to a standard `128x128` pixels.
- **Normalization:** Pixel values are scaled from the `` range to `[^1]`, which helps stabilize training.
- **Data Augmentation:** To prevent overfitting and help the model generalize, the training data is augmented with random rotations, shifts, zooms, and horizontal flips.


### Step 2: Model Architecture

The model combines a pre-trained CNN with a custom attention layer.

1. **CNN Backbone (MobileNetV2):** We use MobileNetV2 pre-trained on ImageNet as a feature extractor. Its weights are "frozen" to leverage its powerful learned features without altering them.
2. **Scaled Dot-Product Attention:** The feature maps from the CNN are fed into a custom attention layer. This layer computes **Query, Key, and Value** vectors to weigh different parts of the image, allowing the model to focus on the most important regions for making a diagnosis.
3. **Classifier Head:** The attention-weighted features are passed to a series of `Dense` layers that perform the final classification.

### Step 3: Training

- **Compilation:** The model is compiled with the `Adam` optimizer, `SparseCategoricalCrossentropy` loss function, and `accuracy` as the evaluation metric.
- **Training Loop:** The model is trained for **50 epochs**, learning from the augmented training data and being validated against the validation set after each epoch.


### Step 4: Evaluation

The model's performance is assessed on the unseen test dataset using several metrics:

- **Overall Accuracy \& Loss:** To get a general sense of performance.
- **Classification Report:** Provides `precision`, `recall`, and `f1-score` for each class.
- **Confusion Matrix:** A visual grid showing what classes the model confuses.
- **Training History Plots:** Visualizing the accuracy and loss curves over epochs helps diagnose overfitting.


## How to Run

This project is designed for Google Colab.

1. Open the `.ipynb` file in Google Colab.
2. Set the runtime to use a **GPU accelerator** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Run the cells sequentially. The `kagglehub` library will handle dataset download and authentication.

## Results

The model was trained for 50 epochs and achieved the following performance on the test set:

- **Test Accuracy:** **66.86%**


### Performance Analysis

- **Training vs. Validation:** The plots of training and validation accuracy/loss show that the model learns well, but some overfitting occurs in later epochs, as the validation accuracy starts to plateau while training accuracy continues to rise.
- **Class-Specific Performance:** The classification report reveals that the model performs best on the `Non-Demented` class. It struggles significantly with the `ModerateDemented` class, mainly due to the severe class imbalance (only 64 test samples for this class).

*(Here, you would insert the images of your plots from the notebook, like the accuracy/loss curves and the confusion matrix).*

## Conclusion \& Future Work

This project successfully implemented a CNN with a self-attention mechanism for Alzheimer's classification. The results are promising but highlight the challenges of working with imbalanced medical imaging datasets.

**Future improvements could include:**

1. **Handling Class Imbalance:** Use techniques like class weighting or oversampling (e.g., SMOTE) to give more importance to minority classes.
2. **Fine-Tuning:** Unfreeze the top layers of MobileNetV2 and train them with a very low learning rate to adapt them better to the MRI data.
3. **Visualizing Attention:** Generate heatmaps to see which parts of the MRI the attention mechanism is focusing on. This would help verify if the model is learning clinically relevant features.



