import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the historical dataset
data = pd.read_csv('C:/Users/ARUL/Downloads/predictive_maintenance (1).csv')

# Feature selection: Choose the features to use for the prediction
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = data[features]
y = data['Failure Type'].apply(lambda x: 1 if x != 'No Failure' else 0)  # Encode failure types

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train a Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Evaluate the Random Forest model
rf_y_pred = rf_model.predict(X_test_scaled)
rf_y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Random Forest Model ---")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred)}")
print("Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=['No Failure', 'Failure']))

# Evaluate the Gradient Boosting model
gb_y_pred = gb_model.predict(X_test_scaled)
gb_y_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Gradient Boosting Model ---")
print(f"Accuracy: {accuracy_score(y_test, gb_y_pred)}")
print("Classification Report:")
print(classification_report(y_test, gb_y_pred, target_names=['No Failure', 'Failure']))

# --- ROC Curve Comparison ---
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_proba)
rf_auc = auc(rf_fpr, rf_tpr)

gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_y_proba)
gb_auc = auc(gb_fpr, gb_tpr)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(gb_fpr, gb_tpr, color='green', lw=2, label=f'Gradient Boosting (AUC = {gb_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# --- Feature Importance ---
plt.figure(figsize=(8, 6))
plt.barh(features, rf_model.feature_importances_, color='skyblue', label='Random Forest')
plt.barh(features, gb_model.feature_importances_, color='orange', alpha=0.7, label='Gradient Boosting')
plt.title('Feature Importance Comparison')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.legend()
plt.show()

# Save models and scaler for future predictions
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Real-time prediction function
def predict_failure(current_data, model_type='random_forest'):
    try:
        # Load the model
        if model_type == 'random_forest':
            model = joblib.load('random_forest_model.pkl')
        elif model_type == 'gradient_boosting':
            model = joblib.load('gradient_boosting_model.pkl')
        else:
            raise ValueError("Invalid model type. Choose 'random_forest' or 'gradient_boosting'.")
        
        # Load the scaler
        scaler = joblib.load('scaler.pkl')

        # Validate input
        if not isinstance(current_data, list) or len(current_data) != len(features):
            raise ValueError("Input data must be a list with the same number of features as the training data.")

        # Prepare input data
        current_data_df = pd.DataFrame([current_data], columns=features)
        current_data_scaled = scaler.transform(current_data_df)

        # Make prediction
        prediction = model.predict(current_data_scaled)
        return "Failure" if prediction[0] == 1 else "No Failure"

    except Exception as e:
        return f"An error occurred: {e}"

# Example of real-time data input
current_input = [900.8, 409, 1805, 41.3, 500]  # Replace with real sensor values
rf_result = predict_failure(current_input, model_type='random_forest')
gb_result = predict_failure(current_input, model_type='gradient_boosting')

print(f"Random Forest Prediction: {rf_result}")
print(f"Gradient Boosting Prediction: {gb_result}")
