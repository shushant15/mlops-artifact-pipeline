import pickle
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, log_loss

# Load the trained model
print("Loading the saved model...")
with open("model_train.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully.")

# Load dataset
print("Loading digits dataset for inference...")
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess data (optional: add scaler if used during training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # NOTE: Fitting again since scaler wasn't saved

# Make predictions
print("Generating predictions...")
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

# Evaluation
print("\nClassification Report:")
print(classification_report(y, y_pred))

loss = log_loss(y, y_proba)
print(f"\nLog Loss: {loss:.4f}")
