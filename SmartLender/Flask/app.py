from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("rdf.pkl", "rb"))
scaler = pickle.load(open("scale1.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():

    gender = int(request.form["Gender"])
    married = int(request.form["Married"])
    dependents = int(request.form["Dependents"])
    education = int(request.form["Education"])
    self_employed = int(request.form["Self_Employed"])
    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_amount_term = float(request.form["Loan_Amount_Term"])
    credit_history = int(request.form["Credit_History"])
    property_area = int(request.form["Property_Area"])

    data = np.array([[

        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area

    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template(
        "submit.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)