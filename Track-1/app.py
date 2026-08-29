import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="ISS AI | Infectious Disease Screening",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_model_package():

    model = joblib.load(
        MODEL_DIR / "xgb_model.joblib"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.joblib"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "label_encoder.joblib"
    )

    metadata = joblib.load(
        MODEL_DIR / "metadata.joblib"
    )

    shap_path = MODEL_DIR / "shap_feature_importance.csv"

    if shap_path.exists():
        shap_importance = pd.read_csv(shap_path)
    else:
        shap_importance = None

    return (
        model,
        preprocessor,
        label_encoder,
        metadata,
        shap_importance
    )


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    (
        model,
        preprocessor,
        label_encoder,
        metadata,
        shap_importance
    ) = load_model_package()

except Exception as e:

    st.error(
        "The screening model could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CUSTOM CSS — DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN APPLICATION
    -------------------------------------------------------- */

    .stApp {
        background-color: #0E1117;
        color: #F5F7FA;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #AEB6C2;
        margin-bottom: 1.2rem;
    }


    /* --------------------------------------------------------
       SECTION HEADINGS
    -------------------------------------------------------- */

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #FFFFFF;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }


    /* --------------------------------------------------------
       RESULT BOX
    -------------------------------------------------------- */

    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #161B22;
        border: 1px solid #30363D;
        margin-bottom: 1rem;
    }

    .result-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-condition {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 0.4rem;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #0B0F14;
        border-right: 1px solid #30363D;
    }

    section[data-testid="stSidebar"] * {
        color: #E6EDF3;
    }


    /* --------------------------------------------------------
       INPUT LABELS
    -------------------------------------------------------- */

    label {
        color: #E6EDF3 !important;
    }


    /* --------------------------------------------------------
       SELECT BOXES
    -------------------------------------------------------- */

    div[data-baseweb="select"] > div {
        background-color: #161B22;
        border-color: #30363D;
    }

    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }


    /* --------------------------------------------------------
       NUMBER INPUTS
    -------------------------------------------------------- */

    input {
        background-color: #161B22 !important;
        color: #FFFFFF !important;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 8px;
        min-height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }


    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 1rem;
        border-radius: 8px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8B949E !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }


    /* --------------------------------------------------------
       DATAFRAMES
    -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #30363D;
        border-radius: 8px;
    }


    /* --------------------------------------------------------
       EXPANDERS
    -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
    }


    /* --------------------------------------------------------
       ALERTS
    -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .small-note {
        color: #8B949E;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ISS AI Hackathon 2026"
    )

    st.markdown(
        "**Track 1**  \n"
        "AI-Assisted Infectious Disease Screening"
    )

    st.divider()

    st.markdown(
        "### About this prototype"
    )

    st.write(
        "This application uses a machine-learning model "
        "to estimate the most likely infectious disease "
        "category from patient information."
    )

    st.write(
        "It is intended to demonstrate an AI-assisted "
        "screening workflow and is not a diagnostic system."
    )

    st.divider()

    st.caption(
        "ISS AI Hackathon 2026"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'AI-Assisted Infectious Disease Screening'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'A machine-learning screening prototype for clinical '
    'assessment support.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the information available from the patient "
    "assessment. The screening tool will return a ranked "
    "estimate across nine infectious disease categories."
)


# ============================================================
# 1. PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '1. Patient information'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    state = st.selectbox(
        "State",
        [
            "Abia",
            "Adamawa",
            "Akwa Ibom",
            "Anambra",
            "Bauchi",
            "Bayelsa",
            "Benue",
            "Borno",
            "Cross River",
            "Delta",
            "Ebonyi",
            "Edo",
            "Ekiti",
            "Enugu",
            "FCT",
            "Gombe",
            "Imo",
            "Jigawa",
            "Kaduna",
            "Kano",
            "Katsina",
            "Kebbi",
            "Kogi",
            "Kwara",
            "Lagos",
            "Nasarawa",
            "Niger",
            "Ogun",
            "Ondo",
            "Osun",
            "Oyo",
            "Plateau",
            "Rivers",
            "Sokoto",
            "Taraba",
            "Yobe",
            "Zamfara"
        ]
    )


with col2:

    month = st.selectbox(
        "Month",
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec"
        ]
    )


with col3:

    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=30
    )


col1, col2, col3 = st.columns(3)


with col1:

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female"
        ]
    )


with col2:

    pregnant = st.selectbox(
        "Pregnant",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col3:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=22.0,
        step=0.1
    )


# ============================================================
# 2. CLINICAL MEASUREMENTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '2. Clinical measurements'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    days_symptoms = st.number_input(
        "Days with symptoms",
        min_value=0,
        max_value=365,
        value=3
    )


with col2:

    temperature_c = st.number_input(
        "Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=37.0,
        step=0.1
    )


with col3:

    heart_rate = st.number_input(
        "Heart rate (bpm)",
        min_value=30,
        max_value=220,
        value=80
    )


col1, col2, col3 = st.columns(3)


with col1:

    resp_rate = st.number_input(
        "Respiratory rate",
        min_value=5,
        max_value=60,
        value=18
    )


with col2:

    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0,
        step=1.0
    )


with col3:

    sbp = st.number_input(
        "Systolic BP (mmHg)",
        min_value=50,
        max_value=250,
        value=120
    )


dbp = st.number_input(
    "Diastolic BP (mmHg)",
    min_value=30,
    max_value=150,
    value=80
)


# ============================================================
# 3. SYMPTOMS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '3. Symptoms'
    '</div>',
    unsafe_allow_html=True
)


symptoms = [
    "fever",
    "cough",
    "sore_throat",
    "headache",
    "vomiting",
    "diarrhea",
    "rash",
    "neck_stiffness",
    "weight_loss",
    "fatigue"
]


symptom_values = {}

cols = st.columns(3)


for i, symptom in enumerate(symptoms):

    with cols[i % 3]:

        symptom_values[symptom] = st.selectbox(
            symptom.replace(
                "_",
                " "
            ).title(),

            [0, 1],

            format_func=lambda x:
            "No" if x == 0 else "Yes",

            key=f"symptom_{symptom}"
        )


# ============================================================
# 4. EXPOSURE AND HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '4. Exposure and history'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    mosquito_exposure = st.selectbox(
        "Mosquito exposure",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col2:

    unsafe_water = st.selectbox(
        "Unsafe water exposure",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col3:

    tb_contact = st.selectbox(
        "TB contact",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


col1, col2, col3 = st.columns(3)


with col1:

    recent_travel = st.selectbox(
        "Recent travel",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col2:

    vaccinated = st.selectbox(
        "Vaccinated",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col3:

    season = st.selectbox(
        "Season",
        [
            "Dry",
            "Rainy"
        ]
    )


# ============================================================
# 5. LABORATORY RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '5. Laboratory results'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    hemoglobin = st.number_input(
        "Hemoglobin",
        min_value=3.0,
        max_value=25.0,
        value=13.0,
        step=0.1
    )


with col2:

    wbc = st.number_input(
        "WBC",
        min_value=0.5,
        max_value=100.0,
        value=7.0,
        step=0.1
    )


with col3:

    platelets = st.number_input(
        "Platelets",
        min_value=10.0,
        max_value=1000.0,
        value=250.0
    )


# ============================================================
# RUN SCREENING
# ============================================================

st.divider()


col1, col2, col3 = st.columns(
    [1, 2, 1]
)


with col2:

    predict_button = st.button(
        "Run AI screening",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Build patient record
    # --------------------------------------------------------

    input_data = {

        "state": state,

        "month": month,

        "age": age,

        "sex": sex,

        "pregnant": pregnant,

        "bmi": bmi,

        "days_symptoms": days_symptoms,

        "temperature_c": temperature_c,

        "heart_rate": heart_rate,

        "resp_rate": resp_rate,

        "spo2": spo2,

        "sbp": sbp,

        "dbp": dbp,

        "fever": symptom_values["fever"],

        "cough": symptom_values["cough"],

        "sore_throat": symptom_values["sore_throat"],

        "headache": symptom_values["headache"],

        "vomiting": symptom_values["vomiting"],

        "diarrhea": symptom_values["diarrhea"],

        "rash": symptom_values["rash"],

        "neck_stiffness": symptom_values["neck_stiffness"],

        "weight_loss": symptom_values["weight_loss"],

        "fatigue": symptom_values["fatigue"],

        "mosquito_exposure": mosquito_exposure,

        "unsafe_water": unsafe_water,

        "tb_contact": tb_contact,

        "recent_travel": recent_travel,

        "vaccinated": vaccinated,

        "season": season,

        "hemoglobin": hemoglobin,

        "wbc": wbc,

        "platelets": platelets
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    # --------------------------------------------------------
    # EXACT MODEL FEATURE ORDER
    # --------------------------------------------------------

    expected_features = [

        "state",
        "month",
        "age",
        "sex",
        "pregnant",
        "bmi",
        "days_symptoms",
        "temperature_c",
        "heart_rate",
        "resp_rate",
        "spo2",
        "sbp",
        "dbp",
        "fever",
        "cough",
        "sore_throat",
        "headache",
        "vomiting",
        "diarrhea",
        "rash",
        "neck_stiffness",
        "weight_loss",
        "fatigue",
        "mosquito_exposure",
        "unsafe_water",
        "tb_contact",
        "recent_travel",
        "vaccinated",
        "season",
        "hemoglobin",
        "wbc",
        "platelets"
    ]


    input_df = input_df[
        expected_features
    ]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        processed_input = preprocessor.transform(
            input_df
        )


        prediction_numeric = model.predict(
            processed_input
        )


        probabilities = model.predict_proba(
            processed_input
        )[0]


        prediction_label = label_encoder.inverse_transform(
            prediction_numeric
        )[0]


        model_probability = float(
            np.max(probabilities)
        )


        # ====================================================
        # SCREENING RESULT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Screening result'
            '</div>',
            unsafe_allow_html=True
        )


        result_col1, result_col2 = st.columns(
            [2, 1]
        )


        with result_col1:

            st.markdown(
                '<div class="result-box">'
                '<div class="result-label">'
                'Most likely condition'
                '</div>'
                '<div class="result-condition">'
                f'{prediction_label}'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        with result_col2:

            st.metric(
                "Model probability",
                f"{model_probability:.1%}"
            )


        st.caption(
            "The probability shown is the model's estimate "
            "for the selected patient profile. It should not "
            "be interpreted as diagnostic certainty."
        )


        # ====================================================
        # RANKED PROBABILITIES
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Screening probabilities'
            '</div>',
            unsafe_allow_html=True
        )


        probability_df = pd.DataFrame({

            "Condition":
                label_encoder.classes_,

            "Probability":
                probabilities
        })


        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        ).reset_index(
            drop=True
        )


        probability_df[
            "Probability (%)"
        ] = (

            probability_df[
                "Probability"
            ] * 100

        ).round(2)


        probability_df.insert(
            0,
            "Rank",
            range(
                1,
                len(probability_df) + 1
            )
        )


        display_df = probability_df[
            [
                "Rank",
                "Condition",
                "Probability (%)"
            ]
        ]


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Relative screening probabilities'
            '</div>',
            unsafe_allow_html=True
        )


        chart_df = probability_df[
            [
                "Condition",
                "Probability (%)"
            ]
        ].set_index(
            "Condition"
        )


        st.bar_chart(
            chart_df
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        if shap_importance is not None:

            st.markdown(
                '<div class="section-title">'
                'What the model learned from the data'
                '</div>',
                unsafe_allow_html=True
            )


            st.write(
                "The features below had the greatest average "
                "influence on model predictions across the "
                "evaluation data. This is a global model "
                "explanation and does not mean that a feature "
                "caused this individual prediction."
            )


            shap_display = shap_importance.copy()


            shap_display.columns = [
                "Feature",
                "Mean absolute SHAP"
            ]


            shap_display = shap_display.head(
                10
            )


            shap_display[
                "Feature"
            ] = (

                shap_display[
                    "Feature"
                ]

                .str.replace(
                    "numeric__",
                    "",
                    regex=False
                )

                .str.replace(
                    "categorical__",
                    "",
                    regex=False
                )

                .str.replace(
                    "_",
                    " ",
                    regex=False
                )

                .str.title()
            )


            st.dataframe(
                shap_display,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # CLINICAL DISCLAIMER
        # ====================================================

        st.warning(
            "Clinical note: This application is an AI-assisted "
            "screening prototype developed for the ISS AI "
            "Hackathon 2026. The output does not establish a "
            "medical diagnosis and should not replace clinical "
            "assessment, laboratory confirmation, or professional "
            "medical judgement."
        )


    except Exception as e:

        st.error(
            "The screening prediction could not be generated."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "ISS AI Hackathon 2026 | Track 1 | "
    "AI-Assisted Infectious Disease Screening"
)