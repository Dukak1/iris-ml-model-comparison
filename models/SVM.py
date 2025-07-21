import pandas as pd

df = pd.read_csv("../data/Iris.csv")

X = df.drop("Species", axis=1)
y = df["Species"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM

from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
target_names = ['setosa', 'versicolor', 'virginica']
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))