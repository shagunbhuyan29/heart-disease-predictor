# Heart Disease Prediction Using Python

This project is a beginner-friendly Machine Learning model that predicts the risk of heart disease using basic health parameters such as blood pressure, heart rate, cholesterol, age, and sex. It uses Python, a simple Logistic Regression model, and a synthetic dataset (heart_dataset.csv).

The project also generates a line chart based on user-inputted values to visually show trends in blood pressure and heart rate.
Features


## Simple heart disease prediction using Logistic Regression

User inputs health data (heart rate, blood pressure, age, etc.)

Line chart visualization of the user's health values

Easy-to-understand dataset

Fully Python-based (beginner friendly)


## Dataset Information

The dataset (heart_dataset.csv) contains the following columm

| Column Name    | Description                             |
| -------------- | --------------------------------------- |
| age            | Age of the person                       |
| sex            | 0 = Female, 1 = Male                    |
| blood_pressure | Resting blood pressure (mm Hg)          |
| heart_rate     | Maximum heart rate                      |
| cholesterol    | Serum cholesterol (mg/dL)               |
| target         | 1 = heart disease, 0 = no heart disease |

The dataset is synthetic but follows realistic medical patterns.

## Model Used

The project uses Logistic Regression, a simple and highly interpretable classification model suitable for beginners and small datasets.

##  Technologies Used

Python 3

Pandas

NumPy

Matplotlib

Scikit-learn

## How to Run the Project
1. Install Required Packages
   
```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Place Files

Ensure the following files are in the same folder:

```css
codes.py (your main Python script)
heart_dataset.csv
```
3. Run the Script
   ```bash
   python codes.py
   ```

Output

The model predicts whether the user is at High Risk or Low Risk.

A line chart is displayed showing user-entered heart rate and blood pressure values.

 Objective

To create a simple and practical ML application that helps beginners understand:

data preprocessing

basic classification models

visualizing health readings

interacting with user input

 Project Structure
│── heart_dataset.csv     # Dataset  
│── codes.py              # Main program (model + line chart)  
│── README.md             # Project documentation  

 Project Description

Heart disease is a major global health concern. This project predicts the possibility of heart disease using basic clinical features. By training a simple model and visualizing trends, it helps understand how risk factors such as heart rate and blood pressure influence predictions.

This is ideal for:

school/college projects

beginners in ML

Python practice
## Author 
## NIHARIKA ADHIRAJSINH DAFLE
