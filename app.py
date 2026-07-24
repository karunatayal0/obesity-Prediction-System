import os
import gradio as gr
import joblib
import pandas as pd

# Load trained model
model = joblib.load("obesity_model.pkl")


def predict_obesity(gender, age, height, weight, family_history, faf, ch2o):
    try:
        # Encode categorical values
        gender = 1 if gender == "Male" else 0
        family_history = 1 if family_history == "Yes" else 0

        # Create dataframe (column names should match training data)
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "family_history_with_overweight": [family_history],
            "FAF": [faf],
            "CH2O": [ch2o]
        })

        prediction = model.predict(input_data)

        return f"Predicted Obesity Level: {prediction[0]}"

    except Exception as e:
        return f"Error: {e}"


demo = gr.Interface(
    fn=predict_obesity,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Height (meters)"),
        gr.Number(label="Weight (kg)"),
        gr.Radio(["Yes", "No"], label="Family History of Obesity"),
        gr.Slider(0, 5, step=1, label="Physical Activity Frequency (FAF)"),
        gr.Slider(1, 3, step=0.5, label="Daily Water Intake (CH2O)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Obesity Prediction System",
    description="Enter the required details to predict obesity level."
)

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
