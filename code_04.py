import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("heart_dataset.csv")

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# -------- USER INPUT SECTION --------
print("\nEnter Patient Details for Prediction:")
age = int(input("Age: "))
sex = int(input("Sex (1=Male, 0=Female): "))
blood_pressure = int(input("Blood Pressure Value: "))
heart_rate = int(input("Heart Rate Value: "))
cholesterol = int(input("Cholesterol Value: "))

# Make a dataframe from user input
user_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "blood_pressure": [blood_pressure],
    "heart_rate": [heart_rate],
    "cholesterol": [cholesterol]
})

# Predict
prediction = model.predict(user_data)[0]
probability = model.predict_proba(user_data)[0][1]  # probability of disease

# Print result
print("\n--- RESULT ---")
if prediction == 1:
    print("Heart Disease: YES")
else:
    print("Heart Disease: NO")

print(f"Probability of Heart Disease: {probability:.2f}")

# -------- LINE CHART OUTPUT --------

# Data for line chart
labels = ["Age", "BP", "Heart Rate", "Cholesterol"]
values = [age, blood_pressure, heart_rate, cholesterol]

plt.plot(labels, values, marker='o')
plt.title("Patient Health Parameters")
plt.xlabel("Parameters")
plt.ylabel("Values")
plt.grid(True)

plt.show()
