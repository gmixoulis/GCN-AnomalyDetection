# Graph Anomaly Detection: Bitcoin Transaction Analysis

## 🕸️ Project Overview

**GCN-AnomalyDetection** applies valid **Graph Convolutional Networks (GCNs)** to identify illicit activities within the Bitcoin network. Using the **Elliptic Data Set**, this project models transactions as a complex graph structure to detect anomalies (money laundering, fraud) with higher accuracy than traditional tabular methods.

## 🔑 Key Features

- **Graph Neural Networks**: Implementation of GCN models (`models.py`) to capture structural dependencies between transactions.
- **Data Pipeline**: Preprocessing scripts (`preprocessing.py`) to structure raw transaction data into graph tensors.
- **Exploratory Analysis**: Deep dive into the dataset characteristics using Jupyter Notebooks.
- **Training Loop**: Custom training logic (`train.py`) optimized for graph-structured data.

## 🛠️ Tech Stack & Skills

- **Language**: Python 3.x
- **ML Frameworks**: PyTorch / TensorFlow (Implied), Scikit-Learn
- **Data Science**: Pandas, NumPy, Matplotlib
- **Domain**: Graph Representation Learning, Financial Forensics

## 💡 Innovation

Detecting financial crime requires looking beyond individual data points. By treating transactions as a **Graph**, this project reveals hidden patterns of collusion and layering that standard classifiers miss, showcasing advanced skills in **Geometric Deep Learning**.

## 📄 Structure

- `train.py`: Main entry point for model training.
- `models.py`: Neural network architecture definitions.
- `preprocessing.py`: Data cleaning and graph construction.
