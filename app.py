import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Agri-Smart: Precision Farming",
    page_icon="🌱",
    layout="wide"
)

# --- LOAD RESOURCES (MODEL & DATA) ---
@st.cache_resource
def load_resources():
    # Define file paths based on your directory tree
    dataset_path = os.path.join('dataset', 'Crop_recommendation.csv')
    model_path = os.path.join('model', 'random_forest_model.joblib')
    
    # 1. Load Dataset
    if not os.path.exists(dataset_path):
        st.error(f"❌ Dataset not found at: {dataset_path}")
        st.stop()
    
    df = pd.read_csv(dataset_path)
    
    # 2. Re-create Label Encoder
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # 3. Load Model (With Fallback for Version Mismatch)
    model = None
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
    except Exception as e:
        # If loading fails (e.g., sklearn version mismatch), we warn and retrain
        st.warning(f"⚠️ Model load failed due to version mismatch. Retraining locally... (Error: {e})")
    
    # If model didn't load (or file missing), Train it now
    if model is None:
        X = df.drop(['label', 'label_encoded'], axis=1)
        y = df['label_encoded']
        
        # Split like in the notebook
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Optional: Save the retrained model so it loads faster next time
        try:
            if not os.path.exists('model'):
                os.makedirs('model')
            joblib.dump(model, model_path)
            st.success("✅ Model retrained and saved locally for compatibility.")
        except:
            pass # Ignore save errors if permission denied

    # 4. Calculate average soil requirements (for the Radar Chart visualization)
    crop_profiles = df.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
    
    return model, le, crop_profiles

# Load resources
try:
    model, le, crop_profiles = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- SIDEBAR: BUSINESS UNDERSTANDING ---
st.sidebar.image("https://img.freepik.com/free-vector/flat-design-agricultural-logo-template_23-2149122393.jpg", width=150)
st.sidebar.title("Agri-Smart")
st.sidebar.markdown("### 🎯 Goal: Zero Hunger (SDG 2)")
st.sidebar.info(
    """
    **Business Value:**
    Farmers often lose money by planting crops incompatible with their soil or overusing fertilizers.
    
    This tool uses **Machine Learning** to:
    1. Recommend the optimal crop for your specific soil conditions.
    2. Provide actionable advice to optimize yield.
    3. Reduce fertilizer waste (SDG 12).
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Created by Kelompok 7")

# --- MAIN PAGE ---
st.title("🌱 Precision Crop Yield & Soil Optimizer")
st.markdown("Enter the soil and weather conditions below to get a scientific recommendation.")

# Input Form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧪 Soil Chemicals")
        n_input = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50, help="Ratio of Nitrogen content in soil")
        p_input = st.number_input("Phosphorous (P)", min_value=0, max_value=145, value=50, help="Ratio of Phosphorous content in soil")
        k_input = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50, help="Ratio of Potassium content in soil")

    with col2:
        st.subheader("🌤️ Environment")
        temp_input = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        hum_input = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
        ph_input = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)

    with col3:
        st.subheader("🌧️ Water")
        rain_input = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)
    
    submit_button = st.form_submit_button("🌱 Predict Best Crop")

# --- PREDICTION LOGIC ---
if submit_button:
    # Prepare input data
    input_data = np.array([[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input]])
    
    # Get Prediction
    prediction_index = model.predict(input_data)[0]
    
    # Convert numerical prediction back to crop name using the LabelEncoder
    predicted_crop = le.inverse_transform([prediction_index])[0]
    
    # Get Probability/Confidence
    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(probabilities) * 100

    # Get Ideal Profile for the predicted crop for comparison
    ideal_profile = crop_profiles.loc[predicted_crop]

    # --- RESULTS SECTION ---
    st.divider()
    
    # Display Result
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.success(f"**Recommended Crop:**")
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{predicted_crop.upper()}</h1>", unsafe_allow_html=True)
        st.metric("Confidence Score", f"{confidence:.2f}%")
        
        # Simple Logic for Actionable Advice
        st.markdown("### 📋 Action Plan")
        advice_list = []
        
        # Check Nitrogen
        if n_input < ideal_profile['N'] - 10:
            advice_list.append(f"⚠️ **Low Nitrogen:** Add urea or organic compost to match ideal level (~{int(ideal_profile['N'])}).")
        elif n_input > ideal_profile['N'] + 20:
            advice_list.append(f"⚠️ **High Nitrogen:** Avoid adding N-fertilizers to prevent runoff.")
            
        # Check Phosphorous
        if p_input < ideal_profile['P'] - 10:
            advice_list.append(f"⚠️ **Low Phosphorous:** Apply phosphate fertilizer.")
            
        # Check Potassium
        if k_input < ideal_profile['K'] - 10:
            advice_list.append(f"⚠️ **Low Potassium:** Add potash fertilizers.")
            
        # Check pH
        if ph_input < ideal_profile['ph'] - 0.5:
            advice_list.append(f"⚠️ **Soil too Acidic:** Consider adding lime.")
        elif ph_input > ideal_profile['ph'] + 0.5:
            advice_list.append(f"⚠️ **Soil too Alkaline:** Consider adding gypsum or sulfur.")

        if not advice_list:
            st.info("✅ Your soil conditions are optimal for this crop!")
        else:
            for advice in advice_list:
                st.warning(advice)

    with res_col2:
        st.subheader("📊 Soil & Climate Compatibility Analysis")
        
        # RADAR CHART LOGIC
        categories = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        # Current Input values
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[n_input, p_input, k_input, temp_input, hum_input, ph_input*10, rain_input/2], # Scaling pH and Rain for visibility
            theta=categories,
            fill='toself',
            name='Your Soil Conditions',
            line_color='blue'
        ))

        fig.add_trace(go.Scatterpolar(
            r=[ideal_profile['N'], ideal_profile['P'], ideal_profile['K'], 
               ideal_profile['temperature'], ideal_profile['humidity'], 
               ideal_profile['ph']*10, ideal_profile['rainfall']/2],
            theta=categories,
            fill='toself',
            name=f'Ideal for {predicted_crop}',
            line_color='green',
            opacity=0.5
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 250] # Fixed range to keep chart stable
                )),
            showlegend=True,
            margin=dict(l=50, r=50, t=30, b=30) # Compact layout
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("*Note: pH is scaled x10 and Rainfall /2 for visual clarity on the chart.")