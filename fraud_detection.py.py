import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
data = pd.read_csv("creditcard.csv")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())
print("\nFraud vs Genuine Transactions:")
print(data['Class'].value_counts())
data['Class'].value_counts().plot(kind='bar')
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()
X = data.drop('Class', axis=1)
y = data['Class']
print("\nFeatures Shape:", X.shape)
print("Labels Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:")
print("Training Features Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
print("\nModel trained successfully!")
predictions = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
joblib.dump(model,'fraud_model.pkl')
print("Model saved successfully!")