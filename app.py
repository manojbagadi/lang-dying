import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ========== PAGE SETUP ==========
st.set_page_config(
    page_title="Vanishing Voices",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗣️ Vanishing Voices")
st.subheader("Language Extinction Risk Predictor")
st.write("Select any language — from major world languages to critically endangered ones — or enter custom data.")

# ========== LOAD ORIGINAL DATA & TRAIN MODEL ==========
@st.cache_data
def load_and_train():
    # Portable path — works no matter where you run the app from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'languages.csv')

    df = pd.read_csv(csv_path)

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

    # Calculate metrics for display
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)

    return df, model, scaler, acc, classification_report(y, y_pred, output_dict=True), confusion_matrix(y, y_pred)

df_endangered, model, scaler, train_acc, class_report, conf_matrix = load_and_train()

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
df_all = pd.concat([df_endangered, df_safe], ignore_index=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio("Input mode:", ["📋 Select from list", "✏️ Enter custom data"])

    st.divider()

    with st.expander("📊 Model Performance"):
        st.metric("Training Accuracy", f"{train_acc*100:.1f}%")
        st.write("**Precision & Recall:**")
        st.write(f"• Not Endangered: P={class_report['0']['precision']:.2f}, R={class_report['0']['recall']:.2f}")
        st.write(f"• Endangered: P={class_report['1']['precision']:.2f}, R={class_report['1']['recall']:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(conf_matrix)

    st.divider()
    st.caption("Built with Streamlit\nUNESCO Endangered Languages + Safe Languages")

# ========== USER INPUT ==========
st.write("### Step 1: Select or Enter Language Data")

if mode == "📋 Select from list":
    language_names = sorted(df_all['Name in English'].tolist())
    selected_name = st.selectbox("Choose a language:", language_names)

    lang_data = df_all[df_all['Name in English'] == selected_name].iloc[0]

    speakers = int(lang_data['Number of speakers'])
    lat = float(lang_data['Latitude'])
    lon = float(lang_data['Longitude'])
    actual_status = str(lang_data['Degree of endangerment'])
    is_safe = (actual_status == "Safe")

else:
    col_a, col_b = st.columns(2)
    with col_a:
        speakers = st.number_input("Number of speakers", min_value=0, value=10000, step=100)
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, format="%.4f")
    with col_b:
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, format="%.4f")
        actual_status = st.selectbox("Known UNESCO Status", 
                                     ["Unknown", "Safe", "Vulnerable", "Definitely endangered", 
                                      "Severely endangered", "Critically endangered", "Extinct"])

    selected_name = "Custom Language"
    is_safe = (actual_status == "Safe")

# ========== LANGUAGE PROFILE ==========
st.write("### Step 2: Language Profile")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🗣️ Language", selected_name)
col2.metric("👥 Speakers", f"{speakers:,}")
col3.metric("📍 Latitude", f"{lat:.4f}")
col4.metric("📍 Longitude", f"{lon:.4f}")

# Edge case: zero speakers
if speakers == 0:
    st.error("⚠️ No recorded speakers. This language may already be extinct.")

# Status badge color
status_colors = {
    "Safe": "🟢",
    "Vulnerable": "🟡",
    "Definitely endangered": "🟠",
    "Severely endangered": "🔴",
    "Critically endangered": "🔴",
    "Extinct": "⚫",
    "Unknown": "⚪"
}
badge = status_colors.get(actual_status, "⚪")
st.info(f"{badge} UNESCO Status: **{actual_status}**")

# ========== MAP ==========
with st.expander("🗺️ View on Map"):
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon], "name": [selected_name]})
    st.map(map_df, zoom=3)

# ========== SIMILAR LANGUAGES ==========
if mode == "📋 Select from list" and not is_safe:
    similar = df_endangered[
        (df_endangered['Number of speakers'] >= speakers * 0.5) & 
        (df_endangered['Number of speakers'] <= speakers * 2) &
        (df_endangered['Name in English'] != selected_name)
    ]
    if not similar.empty:
        with st.expander("🔍 Similar Languages (by speaker count)"):
            st.dataframe(
                similar[['Name in English', 'Number of speakers', 'Degree of endangerment']]
                .sort_values('Number of speakers')
                .head(10),
                use_container_width=True
            )

# ========== SPEAKER COMPARISON CHART ==========
with st.expander("📊 Speaker Count Comparison"):
    fig, ax = plt.subplots(figsize=(8, 4))

    categories = [selected_name, "Median Endangered", "Median Safe"]
    values = [
        speakers,
        df_endangered['Number of speakers'].median(),
        df_safe['Number of speakers'].median()
    ]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Number of Speakers (log scale)")
    ax.set_yscale('log')
    ax.set_ylim(1, max(values) * 2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# ========== SESSION STATE INIT ==========
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.prediction = None
    st.session_state.probability = None
    st.session_state.predicted_label = None

# ========== PREDICTION ==========
st.write("### Step 3: ML Prediction")

if st.button("🔮 Predict Endangerment", type="primary"):
    st.session_state.prediction_done = True

    if is_safe and mode == "📋 Select from list":
        st.session_state.prediction = 0
        st.session_state.probability = [0.0, 0.0]
        st.session_state.predicted_label = "Safe"
    else:
        input_data = np.array([[speakers, lat, lon]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.session_state.prediction = int(prediction)
        st.session_state.probability = probability
        st.session_state.predicted_label = "Endangered" if prediction == 1 else "Not Endangered"

# ========== DISPLAY RESULTS ==========
if st.session_state.prediction_done:

    if st.session_state.predicted_label == "Safe":
        st.success("## 🟢 Safe")
        st.write("This is a major world language with a strong speaker base.")
        st.success(f"✅ {speakers:,} speakers. Not at risk of extinction.")
        st.caption("Note: Safe languages were added manually. The ML model was trained only on at-risk languages.")

    else:
        prediction = st.session_state.prediction
        probability = st.session_state.probability

        conf = probability[1] if prediction == 1 else probability[0]

        if prediction == 1:
            st.error("## 🔴 Endangered")
            st.write(f"**Model Confidence:** {probability[1]*100:.1f}%")
            st.progress(float(probability[1]), text=f"Endangerment Confidence: {probability[1]*100:.1f}%")
            st.write("The ML model predicts this language is at **high risk of extinction**.")
            if speakers < 1000:
                st.warning(f"⚠️ Only {speakers:,} speakers. Languages with < 1,000 speakers are critically vulnerable.")
            elif speakers < 10000:
                st.warning(f"⚠️ Only {speakers:,} speakers. This is a very small community.")
        else:
            st.success("## 🟢 Not Endangered")
            st.write(f"**Model Confidence:** {probability[0]*100:.1f}%")
            st.progress(float(probability[0]), text=f"Safety Confidence: {probability[0]*100:.1f}%")
            st.write("The ML model predicts this language is **relatively safe**.")
            if speakers > 100000:
                st.success(f"✅ Strong speaker base: {speakers:,} people.")

        # Compare prediction vs actual
        st.divider()
        st.write("**Model vs Reality:**")
        actual_simple = "Endangered" if actual_status in ['Critically endangered', 'Severely endangered', 'Definitely endangered'] else "Not Endangered"

        if actual_status == "Unknown":
            st.info(f"ℹ️ Model predicts: **{st.session_state.predicted_label}** (no known status to compare)")
        elif st.session_state.predicted_label == actual_simple:
            st.success(f"✅ Model agrees with reality! Both say: **{actual_simple}**")
        else:
            st.warning(f"⚠️ Model says **{st.session_state.predicted_label}**, but actual status is **{actual_simple}**.")

# ========== FOOTER ==========
st.divider()
st.caption("Built with [Streamlit](https://streamlit.io) | [UNESCO Atlas](https://en.unesco.org/atlas-languages)")