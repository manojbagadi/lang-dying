import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Vanishing Voices", page_icon="🗣️")
st.title("🗣️ Vanishing Voices")
st.subheader("Language Extinction Risk Predictor")
st.write("Select any language — from major world languages to critically endangered ones.")

# ========== LOAD ORIGINAL DATA & TRAIN MODEL ==========
@st.cache_data
def load_and_train():
    df = pd.read_csv('data/languages.csv')
    
    # Keep only what we need
    df = df[['Name in English', 'Number of speakers', 'Latitude', 'Longitude', 'Degree of endangerment']]
    df = df.dropna(subset=['Number of speakers', 'Latitude', 'Longitude', 'Degree of endangerment'])
    
    # Binary target: 1 = Endangered, 0 = Not Endangered
    endangered = ['Critically endangered', 'Severely endangered', 'Definitely endangered']
    df['target'] = df['Degree of endangerment'].apply(lambda x: 1 if x in endangered else 0)
    
    # Train model ONLY on original endangered dataset
    X = df[['Number of speakers', 'Latitude', 'Longitude']]
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return df, model, scaler

df_endangered, model, scaler = load_and_train()

# ========== ADD SAFE LANGUAGES ==========
safe_languages = [
    {"Name in English": "Hindi", "Number of speakers": 600000000, "Latitude": 28.6, "Longitude": 77.2, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "English", "Number of speakers": 1500000000, "Latitude": 51.5, "Longitude": -0.1, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Telugu", "Number of speakers": 83000000, "Latitude": 17.4, "Longitude": 78.5, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Bengali", "Number of speakers": 230000000, "Latitude": 23.8, "Longitude": 90.4, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Marathi", "Number of speakers": 83000000, "Latitude": 19.8, "Longitude": 75.3, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Urdu", "Number of speakers": 70000000, "Latitude": 30.4, "Longitude": 69.3, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Tamil", "Number of speakers": 80000000, "Latitude": 13.1, "Longitude": 80.2, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Spanish", "Number of speakers": 500000000, "Latitude": 40.4, "Longitude": -3.7, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Mandarin Chinese", "Number of speakers": 1100000000, "Latitude": 35.9, "Longitude": 104.2, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Arabic", "Number of speakers": 350000000, "Latitude": 24.7, "Longitude": 46.7, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "French", "Number of speakers": 280000000, "Latitude": 48.9, "Longitude": 2.3, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Portuguese", "Number of speakers": 250000000, "Latitude": 38.7, "Longitude": -9.1, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Russian", "Number of speakers": 260000000, "Latitude": 55.8, "Longitude": 37.6, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Japanese", "Number of speakers": 125000000, "Latitude": 36.2, "Longitude": 138.3, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "German", "Number of speakers": 130000000, "Latitude": 51.2, "Longitude": 9.6, "Degree of endangerment": "Safe", "target": 0},
    {"Name in English": "Korean", "Number of speakers": 80000000, "Latitude": 37.6, "Longitude": 127.0, "Degree of endangerment": "Safe", "target": 0},
]

df_safe = pd.DataFrame(safe_languages)

# Combine both for the dropdown
df_all = pd.concat([df_endangered, df_safe], ignore_index=True)

# ========== USER INPUT ==========
st.write("### Step 1: Select a Language")

language_names = sorted(df_all['Name in English'].tolist())
selected_name = st.selectbox("Choose a language:", language_names)

# Look up selected language
lang_data = df_all[df_all['Name in English'] == selected_name].iloc[0]

speakers = int(lang_data['Number of speakers'])
lat = float(lang_data['Latitude'])
lon = float(lang_data['Longitude'])
actual_status = str(lang_data['Degree of endangerment'])
is_safe = (actual_status == "Safe")

# Show auto-filled data
st.write("### Step 2: Language Profile")
col1, col2, col3 = st.columns(3)
col1.metric("Speakers", f"{speakers:,}")
col2.metric("Latitude", f"{lat:.2f}")
col3.metric("Longitude", f"{lon:.2f}")

st.info(f"📌 UNESCO Status: **{actual_status}**")

# ========== PREDICTION ==========
st.write("### Step 3: ML Prediction")

if st.button("🔮 Predict Endangerment", type="primary"):
    
    # If it's a Safe language we added manually, show directly
    if is_safe:
        st.success("## 🟢 Safe")
        st.write("This is a major world language with a strong speaker base.")
        st.success(f"✅ {speakers:,} speakers. Not at risk of extinction.")
        st.caption("Note: Safe languages were added manually. The ML model was trained only on at-risk languages.")
    
    else:
        # Run ML model for endangered dataset languages
        input_data = np.array([[speakers, lat, lon]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        if prediction == 1:
            st.error("## 🔴 Endangered")
            st.write(f"**Model Confidence:** {probability[1]*100:.1f}%")
            st.write("The ML model predicts this language is at **high risk of extinction**.")
            if speakers < 1000:
                st.warning(f"⚠️ Only {speakers:,} speakers. Languages with < 1,000 speakers are critically vulnerable.")
        else:
            st.success("## 🟢 Not Endangered")
            st.write(f"**Model Confidence:** {probability[0]*100:.1f}%")
            st.write("The ML model predicts this language is **relatively safe**.")
            if speakers > 100000:
                st.success(f"✅ Strong speaker base: {speakers:,} people.")

        # Compare prediction vs actual
        st.divider()
        st.write("**Model vs Reality:**")
        predicted_label = "Endangered" if prediction == 1 else "Not Endangered"
        actual_simple = "Endangered" if actual_status in ['Critically endangered', 'Severely endangered', 'Definitely endangered'] else "Not Endangered"
        
        if predicted_label == actual_simple:
            st.success(f"✅ Model agrees with reality! Both say: **{actual_simple}**")
        else:
            st.warning(f"⚠️ Model says **{predicted_label}**, but actual status is **{actual_simple}**.")

# ========== FOOTER ==========
st.divider()
st.caption("Built with Streamlit | UNESCO Endangered Languages + Safe Languages | ML Mini Project")