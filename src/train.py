import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_config(path):
    print("🔧 Loading hyperparameters from config file...")
    with open(path, 'r') as f:
        return json.load(f)


def train_model(X, y, config):
    print("Training Logistic Regression model...")
    model = LogisticRegression(C=config['C'], solver=config['solver'], max_iter=config['max_iter'])
    model.fit(X, y)
    print("✅ Model training completed.")
    return model


def main():
    print("Starting training pipeline...")
    config = load_config('./config/config.json')
    digits = load_digits()
    X, y = digits.data, digits.target
    print("📊 Dataset loaded successfully. Total samples:", len(X))
    model = train_model(X, y, config)
    with open('model_train.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Trained model saved as model_train.pkl")


if __name__ == '__main__':
    main()
