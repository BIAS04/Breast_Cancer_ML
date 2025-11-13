import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- 1. Load the Model and Scaler ---
try:
    model = joblib.load('LR_Model.pkl')
except FileNotFoundError:
    st.error("Error: 'LR_Model.pkl' file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading LR_Model.pkl: {e}")
    st.stop()

try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: 'scaler.pkl' file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler.pkl: {e}")
    st.stop()

# --- 2. App Title and Page Config ---
st.set_page_config(
    page_title="Cancer Predictor",
    layout="centered"
)
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.write("""
This app predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Not Cancerous)**.
Enter the tumor features below and click 'Predict'.
""")

st.info(
    """
    **Note:** This is a demo project to showcase a Logistic Regression (LR) model. 
    It was trained on a Kaggle dataset (approx. 600 rows, 30 columns) 
    and simplified to use the **top 4 features** for prediction.
    """
)

# --- 3. Input Sliders (Main Window) ---
st.divider()
with st.container(border=True):
    st.subheader("Enter Tumor Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cp_worst = st.slider(
            'Concave Points (Worst)', 
            min_value=0.0, max_value=0.3, value=0.15, step=0.01
        )
        cp_mean = st.slider(
            'Concave Points (Mean)', 
            min_value=0.0, max_value=0.21, value=0.1, step=0.01
        )
    
    with col2:
        p_worst = st.slider(
            'Perimeter (Worst)', 
            min_value=50.0, max_value=260.0, value=100.0, step=1.0
        )
        r_worst = st.slider(
            'Radius (Worst)', 
            min_value=7.0, max_value=37.0, value=15.0, step=0.5
        )

st.divider()

# --- 4. Main Screen Content & Prediction ---

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "Predict Diagnosis", 
        use_container_width=True, 
        type="primary"
    )

results_container = st.container()

if predict_button:
    
    input_features = np.array([[
        cp_worst, 
        p_worst, 
        cp_mean, 
        r_worst
    ]])

    try:
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        with results_container:
            st.header("Prediction Result")
            
            if prediction[0] == 0:
                prob = prediction_proba[0][0]
                st.success("The tumor is **Benign** (Not Cancerous).")
                st.metric(label="Confidence", value=f"{prob * 100:.2f}%")
                
                # --- THIS IS THE NEW ANIMATION ---
                st.balloons()
                # --- END OF UPDATE ---
                
            else:
                prob = prediction_proba[0][1]
                st.error("The tumor is **Malignant** (Cancerous).")
                st.metric(label="Confidence", value=f"{prob * 100:.2f}%")
            
            with st.expander("Show Input Values Used for this Prediction"):
                input_data = {
                    'Feature': [
                        'Concave Points (Worst)', 
                        'Perimeter (Worst)', 
                        'Concave Points (Mean)', 
                        'Radius (Worst)'
                    ],
                    'Input Value': [
                        cp_worst, 
                        p_worst, 
                        cp_mean, 
                        r_worst
                    ]
                }
                input_df = pd.DataFrame(input_data)
                st.dataframe(input_df, use_container_width=True, hide_index=True)

    except ValueError as e:
        st.error(f"An error occurred. This usually means your saved 'scaler.pkl' or 'model.pkl' was not trained on these exact 4 features. Error details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    with results_container:
        st.info("Adjust the sliders above and click 'Predict Diagnosis' to see the result.")