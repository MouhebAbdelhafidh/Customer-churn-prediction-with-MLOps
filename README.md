<h1 align="center">🤖Customer Churn MLOps Pipeline</h1>

Customer Churn MLOps Pipeline
This project implements an end-to-end MLOps pipeline for customer churn prediction (Random forest model).

# 📋Table of Content

## Overview
## Key Features
## Technologies Used
## Usage

## 📜Overview

The pipeline automates all stages of the machine learning lifecycle, from data preparation, model training, and testing to deployment and monitoring. This ensures continuous integration, continuous delivery (CI/CD), and scalability in real-world applications.

## Key Features

### 📚1. Environment Setup

The environment was set up to ensure that all necessary tools and dependencies were installed and configured correctly.

### 📚2. Modularization

The project was modularized by creating a comprehensive model pipeline that covers all stages from data preparation to model evaluation, ensuring a structured approach for each task.

### 📚3. Unit Testing with Pytest

Pytest was used to write and run unit tests, ensuring that each part of the model pipeline was functioning correctly.

### 📚4. CI/CD Integration

Makefile: A Makefile was created to automate common tasks like running tests and training the model. It also includes a file watcher to trigger tasks when relevant code changes.
Airflow: Used to schedule and automate tasks via Directed Acyclic Graphs (DAGs) to ensure continuous execution of pipeline tasks.

### 📚5. Model Deployment with FastAPI

The trained model was deployed using FastAPI, providing a user-friendly interface with Swagger for users to make predictions via an HTTP API.
### 6. Model Monitoring with MLFlow
MLFlow was integrated to track and log model experiments, visualizing the training progress and final model metrics.

### 📚7. Docker Containerization

The model and its dependencies were containerized using Docker, ensuring that the environment is consistent and easily reproducible.

### 📚8. Model Artifacts and Metrics Monitoring

MLFlow artifacts were used to store and track important model outputs.
Elasticsearch was used for logging model metrics, and Kibana was set up to visualize and monitor these logs, ensuring model performance is continuously tracked.

## Technologies Used

🔬Python (for machine learning and pipeline implementation) <br>
🔬Pytest (for unit testing) <br>
🔬FastAPI (for model deployment) <br>
🔬Docker (for containerization) <br>
🔬Airflow (for task scheduling and automation) <br>
🔬MLFlow (for experiment tracking and monitoring) <br>
🔬Elasticsearch & Kibana (for logging and monitoring metrics) <br>

## Usage

1️⃣ Download wsl (Windows Systemfor Linux): wsl --install <br>
2️⃣ Install ubuntu in the wsl: wsl -d Ubuntu-20.04 <br>
3️⃣ Run:  sudo apt update && sudo apt upgrade -y <br>
4️⃣ Install Python 3, pip and virtualenv: sudo apt install python3 python3-pip -y <br>
sudo apt install python3-virtualenv <br>
5️⃣ Clone this repository inside the Ubuntu <br>
6️⃣ Create virtual environnment: virtualenv venv <br>
7️⃣ Activate the virtual invironnment:  source venv/bin/activate <br>
8️⃣ Add execution permissions to the virtual environnment:  chmod +x venv/bin/activate <br>
9️⃣ Run: make help <br>
🔟 Enjoy the trip <br>
