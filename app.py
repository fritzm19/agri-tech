import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Agri-Smart | Precision Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FRONTEND: CUSTOM CSS STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        :root { --primary: #2F855A; --bg-light: #F0FFF4; }
        
        /* Navbar */
        .navbar {
            background-color: white; padding: 1rem 1.5rem; border-bottom: 1px solid #e2e8f0;
            display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;
            border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .brand { font-size: 1.5rem; font-weight: 800; color: var(--primary); display: flex; gap: 10px; }
        
        /* Cards */
        .hero-card {
            background: linear-gradient(135deg, #2F855A 0%, #22543D 100%); color: white;
            padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .result-card {
            background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            text-align: center; border: 1px solid #e2e8f0; height: 100%;
        }
        .crop-name { color: #2F855A; font-size: 3.5rem; font-weight: 800; text-transform: uppercase; margin: 1rem 0; line-height: 1; }
        
        /* Badges */
        .confidence-badge { background-color: #C6F6D5; color: #22543D; padding: 0.5rem 1rem; border-radius: 50px; font-weight: 600; }
        .sdg-badge { background-color: #E9D8FD; color: #553C9A; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-left: 8px; }
        
        /* Advice */
        .advice-box { padding: 12px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95rem; }
        .advice-warning { background-color: #FFF5F5; border-left: 4px solid #FC8181; color: #C53030; }
        .advice-success { background-color: #F0FFF4; border-left: 4px solid #48BB78; color: #276749; }
        
        /* Inputs & Buttons */
        div.stButton > button { background-color: #2F855A; color: white; border-radius: 8px; border: none; padding: 0.75rem; font-weight: 600; width: 100%; }
        div.stButton > button:hover { background-color: #276749; color: white; }
        
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND: ROBUST MODEL LOADING ---
@st.cache_resource
def load_resources():
    dataset_path = os.path.join('dataset', 'Crop_recommendation.csv')
    model_path = os.path.join('model', 'random_forest_model.joblib')
    
    # 1. Load Data
    if not os.path.exists(dataset_path):
        # Fallback for cloud deployment if path varies
        if os.path.exists('Crop_recommendation.csv'):
            dataset_path = 'Crop_recommendation.csv'
        else:
            st.error("Dataset not found. Please upload 'Crop_recommendation.csv'.")
            st.stop()

    df = pd.read_csv(dataset_path)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # 2. Try Loading Model
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except:
            # If version mismatch happens, we just silently ignore it and retrain below
            model = None 

    # 3. Retrain if model is missing or broken (The "One-for-All" Fix)
    if model is None:
        # This block runs automatically if versions don't match
        with st.spinner("Optimizing AI model for current environment..."):
            X = df.drop(['label', 'label_encoded'], axis=1)
            y = df['label_encoded']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Save the fixed model so next run is fast
            try:
                if not os.path.exists('model'): os.makedirs('model')
                joblib.dump(model, model_path)
            except:
                pass # Ignore save errors on read-only cloud filesystems

    # 4. Crop Profiles for Radar Chart
    crop_profiles = df.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
    
    return model, le, crop_profiles

try:
    model, le, crop_profiles = load_resources()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.markdown("## 🌿 Agri-Smart")
    st.markdown("---")
    st.subheader("🧪 Soil Composition")
    n_input = st.number_input("Nitrogen (N)", 0, 140, 90)
    p_input = st.number_input("Phosphorous (P)", 0, 145, 42)
    k_input = st.number_input("Potassium (K)", 0, 205, 43)
    st.subheader("🌤️ Environment")
    temp_input = st.number_input("Temperature (°C)", 0.0, 60.0, 20.8)
    hum_input = st.number_input("Humidity (%)", 0.0, 100.0, 82.0)
    ph_input = st.number_input("pH Level", 0.0, 14.0, 6.5)
    rain_input = st.number_input("Rainfall (mm)", 0.0, 300.0, 202.9)
    st.markdown("---")
    predict_btn = st.button("🚀 Analyze & Recommend", type="primary")

# --- 5. MAIN CONTENT ---
st.markdown("""
    <div class="navbar">
        <div class="brand"><span>🌾 Precision Farming AI</span></div>
        <div>
            <span class="sdg-badge">🎯 SDG 2: Zero Hunger</span>
            <span class="sdg-badge" style="background:#C4F1F9; color:#005585;">♻️ SDG 12: Consumption</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if not predict_btn:
    st.markdown("""
        <div class="hero-card">
            <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;">Intelligent Crop Recommendation</div>
            <div style="font-size: 1.1rem; opacity: 0.9;">
                Input your soil parameters in the sidebar to receive AI-driven crop recommendations tailored to your land's specific conditions.
            </div>
        </div>
    """, unsafe_allow_html=True)

if predict_btn:
    # Prediction
    input_data = np.array([[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input]])
    pred_idx = model.predict(input_data)[0]
    predicted_crop = le.inverse_transform([pred_idx])[0]
    
    # Confidence
    probs = model.predict_proba(input_data)[0]
    confidence = np.max(probs) * 100
    
    # Ideal Profile
    ideal = crop_profiles.loc[predicted_crop]

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown(f"""
            <div class="result-card">
                <div style="color: #718096; text-transform: uppercase; font-weight:600; letter-spacing:1px; font-size:0.9rem;">Best Crop for You</div>
                <div class="crop-name">{predicted_crop}</div>
                <span class="confidence-badge">Match Score: {confidence:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Action Plan")
        advice_html = ""
        
        # Advice Logic
        if n_input < ideal['N'] - 10: advice_html += f"<div class='advice-box advice-warning'>⚠️ <b>Low Nitrogen:</b> Add urea (Target: {int(ideal['N'])})</div>"
        elif n_input > ideal['N'] + 20: advice_html += f"<div class='advice-box advice-warning'>⚠️ <b>High Nitrogen:</b> Reduce fertilizers to prevent runoff.</div>"
        
        if p_input < ideal['P'] - 10: advice_html += f"<div class='advice-box advice-warning'>⚠️ <b>Low Phosphorous:</b> Add phosphate.</div>"
        if k_input < ideal['K'] - 10: advice_html += f"<div class='advice-box advice-warning'>⚠️ <b>Low Potassium:</b> Add potash.</div>"
        
        if ph_input < ideal['ph'] - 0.5: advice_html += f"<div class='advice-box advice-warning'>⚠️ <b>Acidic Soil:</b> Add lime to neutralize.</div>"
        elif ph_input > ideal['ph'] + 0.5: advice_html += f"<div class='advice-box advice-warning'>⚠️ <b>Alkaline Soil:</b> Consider gypsum.</div>"
        
        if advice_html == "": advice_html = "<div class='advice-box advice-success'>✅ Conditions are optimal! No action needed.</div>"
        st.markdown(advice_html, unsafe_allow_html=True)

    with col2:
        # Chart
        categories = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temp', 'Humidity', 'pH', 'Rain']
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[n_input, p_input, k_input, temp_input*2, hum_input, ph_input*10, rain_input/2],
            theta=categories, fill='toself', name='Your Soil', line_color='#3182CE'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[ideal['N'], ideal['P'], ideal['K'], ideal['temperature']*2, ideal['humidity'], ideal['ph']*10, ideal['rainfall']/2],
            theta=categories, fill='toself', name=f'Ideal {predicted_crop}', line_color='#48BB78', opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 250])),
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
            title="Soil Profile Gap Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)