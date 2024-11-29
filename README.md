
# Phishing Website Detection with Machine Learning

This project demonstrates how to classify URLs as malicious or benign using a Random Forest classifier. The code uses the UCI Machine Learning Repository's dataset for training and evaluation and applies the trained model to a custom dataset.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [License](#license)

## Overview
This repository contains Python code to:
1. Train a machine learning model (Random Forest) using phishing data from the UCI repository.
2. Evaluate the model on both the UCI dataset and a custom dataset (`balanced_urls.csv`).
3. Visualize key results, including feature importance, confusion matrices, and class distributions.

The features used for classification include URL characteristics like the presence of an IP address, length, HTTPS usage, and more.

## Requirements
- Python 3.8 or later
- Libraries:
  - `ucimlrepo`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/phishing-detection.git
   cd phishing-detection
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `balanced_urls.csv` file is placed in the root directory of the project.

## Usage
1. Run the script to train and evaluate the model:
   ```bash
   python phishing_detection.py
   ```

2. The script fetches the UCI dataset, preprocesses the data, trains a Random Forest classifier, and evaluates it on both datasets.

## Features
- **Feature Extraction**:
  - Extracts key features from URLs (e.g., presence of IP address, URL length, HTTPS usage).
  - Adds placeholder values for features not derivable from raw URLs.

- **Model Training**:
  - Uses a Random Forest classifier from scikit-learn.

- **Evaluation**:
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score

- **Custom Dataset**:
  - Handles and preprocesses a custom dataset (`balanced_urls.csv`) for evaluation.

## Evaluation Metrics
The model's performance is evaluated using:
1. **Confusion Matrix**: Shows the number of true positives, true negatives, false positives, and false negatives.
2. **Classification Report**: Provides precision, recall, F1-score, and support for each class.
3. **ROC-AUC Score**: Measures the model's ability to distinguish between classes.

## Visualizations
The script includes visualizations for better understanding of results:
1. **Feature Importance**: Displays the importance scores of each feature in the Random Forest model.
2. **Confusion Matrices**: Heatmaps for both UCI and custom datasets.
3. **ROC Curve**: Plots the ROC curve for the UCI dataset.
4. **Class Distributions**: Bar plots showing the distribution of classes in the datasets.

### Example Visualizations
#### Feature Importance
A bar plot showing which features most influence the model's decisions.

#### Confusion Matrix
Heatmaps for predicted vs actual classifications for both datasets.

#### ROC Curve
Graphical representation of the model's true positive rate vs false positive rate.

#### Class Distribution
Bar charts showing the number of samples for each class in the datasets.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
