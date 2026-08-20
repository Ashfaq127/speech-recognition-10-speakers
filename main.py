import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_evaluate_model():
    data_path = "./data/processed/processed_data.npz"
    if not os.path.exists(data_path):
        print("[ERROR] Processed data not found! Run dataset_processor.py first.")
        return
        
    print("[INFO] Loading preprocessed MFCC matrices from dataset_processor.py...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"[INFO] Loaded {X.shape[0]} samples with feature dimensions: {X.shape[1]}")
    
    # Split dataset into 80% training and 20% validation sets with class stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[INFO] Constructing Voice Identification Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("[INFO] Training Model...")
    model.fit(X_train, y_train)
    
    # Generate predictions and calculate model accuracy
    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_test, y_pred)
    
    print(f"[SUCCESS] Training Complete.")
    print(f" -> Training Accuracy: {train_acc * 100:.2f}%")
    print(f" -> Validation Accuracy: {val_acc * 100:.2f}%")
    
    print("\n[EVALUATION REPORT] Classification Detail:")
    speakers = [f"Speaker_{i:02d}" for i in range(1, 11)]
    print(classification_report(y_test, y_pred, target_names=speakers))
    
    print("\n[EVALUATION REPORT] Confusion Matrix Generated:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained model pipeline for future prediction tasks
    joblib.dump(model, 'speaker_identification_model.pkl')
    print("\n[SAVED] Model saved successfully as speaker_identification_model.pkl")

if __name__ == "__main__":
    train_and_evaluate_model()