import os
import joblib
import gradio as gr
import pandas as pd

# Load model
model = joblib.load("obesity_prediction_model.pkl")

# Mappings
gender = {"Male": 0, "Female": 1}
yes_no = {"no": 0, "yes": 1}
calc = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4,
}

labels = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III",
}


def predict(age, gender_, height, weight, calc_, favc, fcvc,
            smoke, ch2o, family, faf, tue, transport):

    data = pd.DataFrame([[
        age,
        gender[gender_],
        height,
        weight,
        calc[calc_],
        yes_no[favc],
        fcvc,
        yes_no[smoke],
        ch2o,
        yes_no[family],
        faf,
        tue,
        mtrans[transport]
    ]], columns=[
        "Age", "Gender", "Height", "Weight", "CALC", "FAVC",
        "FCVC", "SMOKE", "CH2O",
        "family_history_with_overweight",
        "FAF", "TUE", "MTRANS"
    ])

    pred = model.predict(data)[0]

    try:
        pred = int(pred)
        return f"## 🩺 Prediction\n### {labels[pred]}"
    except:
        return f"## 🩺 Prediction\n### {pred}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(label="Height (m)"),
        gr.Number(label="Weight (kg)"),
        gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], label="Alcohol Consumption"),
        gr.Dropdown(["no", "yes"], label="High Calorie Food"),
        gr.Slider(1, 3, step=1, label="Vegetable Consumption"),
        gr.Dropdown(["no", "yes"], label="Smoking"),
        gr.Slider(1, 3, step=0.5, label="Water Intake"),
        gr.Dropdown(["no", "yes"], label="Family History"),
        gr.Slider(0, 3, step=0.5, label="Physical Activity"),
        gr.Slider(0, 2, step=0.5, label="Technology Usage"),
        gr.Dropdown(
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
            label="Transportation"
        ),
    ],
    outputs=gr.Markdown(),
    title="🏥 Obesity Level Prediction",
    description="Enter your details to predict obesity level using a KNN Machine Learning model.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
