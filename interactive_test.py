import pandas as pd
import joblib

# Load the saved model
model = joblib.load('fraud_model.pkl')
print("Model loaded successfully!\n")

# Load the dataset
data = pd.read_csv("creditcard.csv")

while True:
    # Ask user for a row number
    user_input = input(f"Enter a row number (0 to {len(data)-1}), or 'q' to quit: ")
    
    if user_input.lower() == 'q':
        print("Exiting...")
        break
    
    try:
        row_num = int(user_input)
        if row_num < 0 or row_num >= len(data):
            print("Invalid row number. Try again.\n")
            continue
    except ValueError:
        print("Please enter a valid number.\n")
        continue
    
    # Get that transaction
    transaction = data.iloc[[row_num]].drop('Class', axis=1)
    actual_label = data.iloc[row_num]['Class']
    
    # Predict
    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0]
    
    print(f"\nTransaction Amount: {data.iloc[row_num]['Amount']}")
    print(f"Model Prediction: {'FRAUD' if prediction == 1 else 'GENUINE'}")
    print(f"Confidence: {max(probability)*100:.2f}%")
    print(f"Actual Label: {'FRAUD' if actual_label == 1 else 'GENUINE'}")
    print("-" * 40, "\n")