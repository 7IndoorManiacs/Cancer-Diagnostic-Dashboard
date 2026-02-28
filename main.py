import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. LOAD & PREPARE DATA
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['diagnosis'] = cancer_data.target
print("✅ Step 1: Data Loaded!")

# 2. SCALE & SPLIT
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("✅ Step 2: Data Squished and Split!")

# 3. TRAIN THE MODEL
model = LogisticRegression()
model.fit(X_train, y_train)
print("✅ Step 3: Robot Brain Trained!")

# 4. SAVE THE BRAINS
if not os.path.exists('static'):
    os.makedirs('static')
joblib.dump(model, 'cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Brain and Squisher saved as .pkl files!")

# 5. GENERATE & SAVE THE MISTAKE MAP (Confusion Matrix)
plt.figure(figsize=(8, 6))
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sick', 'Healthy'])
disp.plot(cmap='Blues')
plt.title("The 'Mistake Map' (Confusion Matrix)")
plt.savefig('static/confusion_matrix.png') # <--- Save separately
plt.close() # <--- Clear the drawing board!

# 6. GENERATE & SAVE THE S-CURVE
X_single = X_test[:, 0].reshape(-1, 1)
model_simple = LogisticRegression().fit(X_single, y_test)
X_range = np.linspace(X_single.min(), X_single.max(), 300).reshape(-1, 1)
y_prob = model_simple.predict_proba(X_range)[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_single, y_test, color='purple', alpha=0.5, label='Actual Cells')
plt.plot(X_range, y_prob, color='red', linewidth=3, label='The S-Curve')
plt.axhline(0.5, color='black', linestyle='--', label='50% Threshold')
plt.title("The Invisible 'S-Curve' of Your Model")
plt.legend()
plt.savefig('static/scurve.png') # <--- Save separately
plt.close() # <--- Clear the drawing board!

print("✅ Step 4: Visualizations Saved to /static folder!")

# 7. FINAL REPORT CARD
score = accuracy_score(y_test, predictions)
print(f"\n--- 🎓 FINAL REPORT CARD ---")
print(f"Accuracy: {score * 100:.2f}%")