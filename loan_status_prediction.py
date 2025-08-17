import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pickle

with open('loan_status_prediction_model.pkl', 'rb') as f:
    pca, model, scaler = pickle.load(f)

feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 
                 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

def predict_loan():
    try:
        gender = int(gender_var.get())
        married = int(married_var.get())
        dependents = int(dependents_var.get())
        education = int(education_var.get())
        self_employed = int(self_employed_var.get())
        applicant_income = float(applicant_income_var.get())
        coapplicant_income = float(coapplicant_income_var.get())
        loan_amount = float(loan_amount_var.get())
        loan_amount_term = float(loan_amount_term_var.get())
        credit_history = float(credit_history_var.get())
        property_area = int(property_area_var.get())

        input_data = pd.DataFrame([[gender, married, dependents, education, self_employed, applicant_income, 
                                    coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]],
                                  columns=feature_names)

        scaled_data = scaler.transform(input_data)

        scaled_data = pd.DataFrame(scaled_data, columns=feature_names)

        input_data_pca = pca.transform(scaled_data)

        prediction = model.predict(input_data_pca)

        if prediction == 1:
            messagebox.showinfo("Prediction Result", "Loan Approved!")
        else:
            messagebox.showinfo("Prediction Result", "Loan Denied!")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Loan Status Prediction")

tk.Label(root, text="Gender (1: Male, 0: Female)").grid(row=0, column=0)
gender_var = tk.StringVar()
tk.Entry(root, textvariable=gender_var).grid(row=0, column=1)

tk.Label(root, text="Married (1: Yes, 0: No)").grid(row=1, column=0)
married_var = tk.StringVar()
tk.Entry(root, textvariable=married_var).grid(row=1, column=1)

tk.Label(root, text="Dependents (0, 1, 2, or 3+)").grid(row=2, column=0)
dependents_var = tk.StringVar()
tk.Entry(root, textvariable=dependents_var).grid(row=2, column=1)

tk.Label(root, text="Education (1: Graduate, 0: Not Graduate)").grid(row=3, column=0)
education_var = tk.StringVar()
tk.Entry(root, textvariable=education_var).grid(row=3, column=1)

tk.Label(root, text="Self Employed (1: Yes, 0: No)").grid(row=4, column=0)
self_employed_var = tk.StringVar()
tk.Entry(root, textvariable=self_employed_var).grid(row=4, column=1)

tk.Label(root, text="Applicant Income in Dollars").grid(row=5, column=0)
applicant_income_var = tk.StringVar()
tk.Entry(root, textvariable=applicant_income_var).grid(row=5, column=1)

tk.Label(root, text="Coapplicant Income in Dollars").grid(row=6, column=0)
coapplicant_income_var = tk.StringVar()
tk.Entry(root, textvariable=coapplicant_income_var).grid(row=6, column=1)

tk.Label(root, text="Loan Amount (eg. 100 = 100K Dollars)").grid(row=7, column=0)
loan_amount_var = tk.StringVar()
tk.Entry(root, textvariable=loan_amount_var).grid(row=7, column=1)

tk.Label(root, text="Loan Amount Term in Months(eg., 360)").grid(row=8, column=0)
loan_amount_term_var = tk.StringVar()
tk.Entry(root, textvariable=loan_amount_term_var).grid(row=8, column=1)

tk.Label(root, text="Credit History (1 or 0)").grid(row=9, column=0)
credit_history_var = tk.StringVar()
tk.Entry(root, textvariable=credit_history_var).grid(row=9, column=1)

tk.Label(root, text="Property Area (1: Urban, 2: Semiurban, 3: Rural)").grid(row=10, column=0)
property_area_var = tk.StringVar()
tk.Entry(root, textvariable=property_area_var).grid(row=10, column=1)

predict_button = tk.Button(root, text="Predict Loan Status", command=predict_loan)
predict_button.grid(row=11, columnspan=2)

root.mainloop()
