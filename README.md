# MLOps Artifact Pipeline

## Project Overview

This project implements a complete MLOps pipeline using GitHub Actions for a digit classification task using Logistic Regression. The pipeline includes model training, unit testing, and inference, all automated using CI/CD workflows.

## Objectives

- Train a Logistic Regression model on the `digits` dataset from `sklearn`.
- Use a JSON config file to load hyperparameters.
- Save the trained model as an artifact.
- Write unit tests to validate training logic and configuration.
- Set up GitHub Actions workflows for:
  - Model Training
  - Testing using Pytest
  - Inference with job dependencies
- Pass model artifacts between jobs using `upload-artifact` and `download-artifact`.

## Directory Structure

.
├── .github/
│   └── workflows/
│       ├── train.yml
│       ├── test.yml
│       └── inference.yml
├── config/
│   └── config.json
├── src/
│   ├── train.py
│   └── inference.py
├── tests/
│   └── test_train.py
├── .gitignore
├── model_train.pkl
├── README.md
└── requirements.txt

## Setup Instructions

1. Create and activate a conda environment:
   conda create -n mlops-env python=3.10 -y  
   conda activate mlops-env

2. Install dependencies:
   pip install -r requirements.txt

3. Run training locally:
   python3 src/train.py

4. Run unit tests locally:
   python3 -m pytest tests/test_train.py

5. Run inference locally:
   python3 src/inference.py

## GitHub Actions Workflows

- train.yml: Trains the model and uploads `model_train.pkl` as an artifact.
- test.yml: Runs unit tests using `pytest`.
- inference.yml: Executes tests → training → inference in sequence using job dependencies and artifact passing.

## Branching Strategy

- main: Initial setup with README
- classification_branch: Implements training logic
- test_branch: Adds unit tests
- inference_branch: Adds inference and multi-job workflow

Each branch builds on the previous one to maintain clean history and workflow integrity.

## Output Artifact

The final trained model is stored as:
model_train.pkl

It is passed between GitHub Actions jobs using the official actions/upload-artifact and actions/download-artifact.

## Author

Name: Shushant Kumar Tiwari  
Roll No: G24AI1116  
Repo: https://github.com/shushant15/mlops-artifact-pipeline
