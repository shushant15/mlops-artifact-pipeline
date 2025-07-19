import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model

def test_config_load():
    config = load_config('./config/config.json')
    assert 'C' in config and isinstance(config['C'], float)
    assert 'solver' in config and isinstance(config['solver'], str)
    assert 'max_iter' in config and isinstance(config['max_iter'], int)
    print("Config file loaded successfully.")

def test_model_training():
    print("Verifying if model trains properly...")
    config = load_config('./config/config.json')
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_')
    print("✅ Model is trained and has learned coefficients.")

def test_model_accuracy():
    print("📈 Checking model accuracy...")
    config = load_config('./config/config.json')
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    acc = model.score(X, y)
    print(f"🔍 Accuracy on training data: {acc:.4f}")
    assert acc > 0.9
    print("☑️ Accuracy is above 90%.")
