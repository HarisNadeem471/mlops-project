# src/train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model.pkl')

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
