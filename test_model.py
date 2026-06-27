import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load the saved model
model = joblib.load('fraud_model.pkl')
print("Model loaded successfully!")

# Load the dataset again (just to get some test samples)
data = pd.read_csv("creditcard.csv")
X = data.drop('Class', axis=1)
y = data['Class']

# Use the same split as before so it's a fair test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test on a few samples
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("\nPrediction for first test row:", prediction)
print("Actual label for first test row:", y_test.iloc[0])

# Test on multiple rows
sample_multi = X_test.iloc[0:5]
predictions_multi = model.predict(sample_multi)

print("\nPredictions for first 5 rows:", predictions_multi)
print("Actual labels for first 5 rows:", y_test.iloc[0:5].values)