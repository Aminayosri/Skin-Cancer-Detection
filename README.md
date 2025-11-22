# Skin-Cancer-Detection
This project uses a CNN (Keras/TensorFlow) on the HAM10000 dataset (10,000+ images) for 7-class skin lesion classification. Key techniques include Augmentation and Dropout. The system achieved 98.5% Accuracy on the test set. Development used an Agile (2-week) sprint methodology.
That's a very comprehensive project plan! Here is a structured Description and README based on your provided information, formatted for a GitHub reposi


🌟 Overview

This repository contains the code and documentation for the Skin Cancer Detection Project, an initiative to create a highly accurate system for classifying skin lesions using Deep Learning. The project utilizes the HAM10000 dataset and a custom-designed Convolutional Neural Network (CNN) to distinguish between seven dermatological categories.

🎯 Objectives

Model Development: Build a high-performance CNN for 7-class image classification.

Custom Preprocessing: Implement resizing (to $128 \times 128$ RGB) and normalization ($[0, 1]$ range) for consistent input data.

Data Augmentation: Apply techniques (flipping, rotation, zooming) to enhance model generalization and address class imbalance.

Evaluation: Achieve high accuracy and provide comprehensive visualization of model performance (Confusion Matrices, Training Plots).

📊 Dataset: HAM10000

Name: HAM10000 (Human Against Machine with 10000 training images)

Volume: Over 10,000 dermoscopic images of skin lesions.

Classes (7): Includes common benign lesions (e.g., Benign keratosis-like lesions) and serious conditions (e.g., Melanoma).

Format: JPG images with corresponding labels in HAM10000_metadata.csv.

Preprocessing Summary

Resizing: All images standardized to $128 \times 128$ pixels (RGB).

Normalization: Pixel values scaled to $[0, 1]$.

Data Split: 80% Training (stratified), 20% Testing.

🧠 System Architecture: CNN Model

The classification model is a deep Convolutional Neural Network built with the Keras Sequential API.

Layer TypeConfigurationPurposeInput Layer$128 \times 128 \times 3$ (RGB)Accepts preprocessed images.Convolutional BlocksConv2D (3x3), ReLU, Batch Normalization, Max PoolingExtracts increasingly complex spatial features.Deep Feature ExtractionIncreasing Filters (32, 64, 128, 256)Allows the model to learn hierarchical patterns.FlattenMulti-dim $\rightarrow$ 1D vectorPrepares data for fully connected layers.Fully Connected Layers256, 128, 64, 32 Neurons (ReLU)High-level non-linear feature combination.RegularizationDropout (Rate 0.2), L1L2 RegularizationMitigates overfitting and improves generalization.Output Layer7 Neurons (Softmax Activation)Outputs probability for each of the 7 lesion classes.Model Compilation Details

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

✅ Evaluation and Results

The model was rigorously tested on the unseen 20% test set.

MetricResultTest Accuracy98.5%Test LossDocumented in training logsKey Observations

Generalization: Data augmentation techniques significantly improved the model's ability to generalize to new data.

Challenge: Class imbalance slightly impacted the performance for the rarest lesion categories, which was partially mitigated by augmentation.

⚙️ Agile Implementation (Two-Week Timeline)

SprintWeekTasksDeliverables1: Data Prep & Exploration1Load, inspect, and visualize data; Implement preprocessing (resizing, normalization); Establish data augmentation pipeline.Fully preprocessed dataset; Augmentation pipeline; Class distribution visualizations.2: Model Dev & Evaluation2Design CNN; Train model; Evaluate performance; Visualize metrics; Refine hyperparameters.Fully trained CNN model (.h5 or equivalent); Documented metrics; Comprehensive final report.👥 Team & Roles

Team MemberRole: Nada >> Data Preprocessing and Augmentation, Amira Reda >> Model Development and Training and >> Amina Yosri Validation, Visualization, and Documentation

📚 References
HAM10000 Dataset: https://doi.org/10.1038/sdata.2018.161
TensorFlow Documentation: https://www.tensorflow.org
