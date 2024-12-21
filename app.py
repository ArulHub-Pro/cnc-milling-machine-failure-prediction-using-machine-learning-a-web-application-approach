from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Retrieve form inputs
            air_temp = float(request.form["Air temperature [K]"])
            process_temp = float(request.form["Process temperature [K]"])
            rot_speed = float(request.form["Rotational speed [rpm]"])
            torque = float(request.form["Torque [Nm]"])
            tool_wear = float(request.form["Tool wear [min]"])

            # Prepare data for prediction
            input_data = pd.DataFrame([[air_temp, process_temp, rot_speed, torque, tool_wear]],
                                      columns=["Air temperature [K]", "Process temperature [K]",
                                               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)
            result = "Failure" if prediction[0] == 1 else "No Failure"
        except Exception as e:
            result = f"Error: {e}"

    # Render the HTML template and pass the result
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
