import os
import joblib
import pandas as pd
import gradio as gr

# Load model
model = joblib.load("obesity_model.pkl")

# Encoding dictionaries
gender_map = {
    "Male": "Male",
    "Female": "Female"
}

yes_no = {
    "Yes": "yes",
    "No": "no"
}

calc_map = {
    "Never": "no",
    "Sometimes": "Sometimes",
    "Frequently": "Frequently",
    "Always": "Always"
}

transport_map = {
    "Walking": "Walking",
    "Bike": "Bike",
    "Motorbike": "Motorbike",
    "Public Transportation": "Public_Transportation",
    "Automobile": "Automobile"
}


def predict(age, gender, height, weight, calc, favc,
            fcvc, ncp, smoke, ch2o,
            family_history, faf, tue, transport):

    try:

        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender_map[gender]],
            "Height": [height],
            "Weight": [weight],
            "CALC": [calc_map[calc]],
            "FAVC": [yes_no[favc]],
            "FCVC": [fcvc],
            "NCP": [ncp],
            "SMOKE": [yes_no[smoke]],
            "CH2O": [ch2o],
            "family_history_with_overweight": [yes_no[family_history]],
            "FAF": [faf],
            "TUE": [tue],
            "MTRANS": [transport_map[transport]]
        })

        prediction = model.predict(data)

        return f"Predicted Obesity Class: {prediction[0]}"

    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Height (m)"),
        gr.Number(label="Weight (kg)"),
        gr.Dropdown(["Never", "Sometimes", "Frequently", "Always"], label="Alcohol Consumption"),
        gr.Radio(["Yes", "No"], label="High Calorie Food"),
        gr.Slider(1, 3, step=0.5, label="Vegetable Consumption (FCVC)"),
        gr.Slider(1, 4, step=1, label="Main Meals (NCP)"),
        gr.Radio(["Yes", "No"], label="Smoking"),
        gr.Slider(1, 3, step=0.5, label="Water Intake (CH2O)"),
        gr.Radio(["Yes", "No"], label="Family History"),
        gr.Slider(0, 3, step=1, label="Physical Activity (FAF)"),
        gr.Slider(0, 2, step=0.5, label="Technology Usage (TUE)"),
        gr.Dropdown(
            [
                "Walking",
                "Bike",
                "Motorbike",
                "Public Transportation",
                "Automobile"
            ],
            label="Transportation"
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Obesity Prediction System"
)

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
