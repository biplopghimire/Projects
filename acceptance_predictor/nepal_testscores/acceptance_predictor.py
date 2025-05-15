import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np

orignal_data = pd.read_csv("Sample_College_Applicants_Dataset.csv")

# print("Original Data:")
# print(orignal_data.head())

orignal_data["Acceptance_Status"] = orignal_data["Acceptance_Status"].map({"Rejected": 0, "Accepted": 1})
processed_data = pd.get_dummies(orignal_data, columns=["Gender", "Program_Applied"], drop_first=True)

# print("\nProcessed Data:")
# print(processed_data.head())

from sklearn.model_selection import train_test_split

# Split features and target
X = processed_data.drop(["Acceptance_Status", "ApplicantID"], axis=1)
y = processed_data["Acceptance_Status"]

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy: {accuracy:.2f}")


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

tree_pred = tree_clf.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_pred)
print(f"\nDecision Tree Accuracy: {tree_accuracy:.2f}")



def predict_acceptance(age, ielts_score, highschool_percentage, gender, program_applied):
    input_dict = {
        "Age": age,
        "IELTS_Score": ielts_score,
        "HighSchool_Percentage": highschool_percentage,
        "Gender_Male": 0,
        "Gender_Other": 0,
        "Program_Applied_Business": 0,
        "Program_Applied_Early Childhood Education": 0,
        "Program_Applied_Engineering": 0,
        "Program_Applied_Health Sciences": 0,
        "Program_Applied_Hospitality": 0,
        "Program_Applied_IT": 0
    }

    if gender == "Male":
        input_dict["Gender_Male"] = 1
    elif gender == "Other":
        input_dict["Gender_Other"] = 1

    program_key = f"Program_Applied_{program_applied}"
    if program_key in input_dict:
        input_dict[program_key] = 1

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    prediction_log = logreg.predict(input_df)[0]
    probability_log = logreg.predict_proba(input_df)[0][1]
    label_log = "Accepted" if prediction_log == 1 else "Rejected"


    prediction_tree = tree_clf.predict(input_df)[0]
    label_tree = "Accepted" if prediction_tree == 1 else "Rejected"
    return f"Tree: {label_tree}" + f" | Log: {label_log} (Probability: {probability_log:.2%})"

print("\nPredicting acceptance for a new student:")

# Example input
print(predict_acceptance(age=21, ielts_score=6.9, highschool_percentage=70, gender="Male", program_applied="Health Sciences"))