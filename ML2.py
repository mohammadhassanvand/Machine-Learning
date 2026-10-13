
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load dataset
data_path = Path(__file__).resolve().parent / "titanic.csv"
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)

# 2. Analyze survival by gender
survival_by_sex = df.groupby("Sex")["Survived"].mean() * 100

survival_by_sex.plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Analyze survival by passenger class
survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

survival_by_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Prepare features and target
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features].copy()
y = df["Survived"]

# 5. Handle missing values
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# 6. Convert categorical data into numeric data
X = pd.get_dummies(X, columns=["Sex", "Embarked"], dtype=int)

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate Decision Tree
print("\nModel Evaluation")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar", figsize=(9, 5))
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 12. Compare actual and predicted results
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))

# 13. Train Logistic Regression
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nLogistic Regression Results")
print("Accuracy:", logistic_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions))

# Compare both models
decision_tree_accuracy = accuracy_score(y_test, y_pred)

print("\nModel Comparison")
print("Decision Tree Accuracy:", decision_tree_accuracy)
print("Logistic Regression Accuracy:", logistic_accuracy)

# 14. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Did not survive", "Survived"],
    cmap="Blues"
)

plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

# 15. Save the trained model
model_path = Path(__file__).resolve().parent / "titanic_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

# 16. Visualize model comparison
model_names = ["Decision Tree", "Logistic Regression"]
accuracies = [decision_tree_accuracy, logistic_accuracy]

plt.figure(figsize=(7, 5))
plt.bar(model_names, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# 17. Load the saved model

loaded_model = joblib.load(model_path)

# Predict using the loaded model
sample = X_test.iloc[[0]]
predictions = loaded_model.predict(sample)[0]

print("\nSaved Model Prediction")
print("Predicted survival:", predictions)

if predictions == 1:
    print("The model predicts that the passenger survived.")
else:
    print("The model predicts that the passenger did not survive.")

print("Actual result:", y_test.iloc[0])