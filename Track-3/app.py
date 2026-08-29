import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ISS AI | Climate & Environmental Health",
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
# LOAD TRACK 3 MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_model_package():

    pipeline = joblib.load(
        MODEL_DIR / "track3_xgboost_pipeline.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "track3_label_encoder.pkl"
    )

    feature_names = joblib.load(
        MODEL_DIR / "track3_feature_names.pkl"
    )

    metadata = joblib.load(
        MODEL_DIR / "track3_model_metadata.pkl"
    )

    return (
        pipeline,
        label_encoder,
        feature_names,
        metadata
    )


try:

    (
        pipeline,
        label_encoder,
        feature_names,
        metadata
    ) = load_model_package()

except Exception as e:

    st.error(
        "Unable to load the Track 3 model package."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #0E1117;
        color: #F5F7FA;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    p {
        color: #D0D7DE;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #9DA7B3;
        margin-bottom: 1.25rem;
    }


    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #FFFFFF;
        margin-top: 1.25rem;
        margin-bottom: 0.85rem;
    }


    /* ======================================================
       SECTION CARDS
       ====================================================== */

    .section-card {
        background-color: #151A21;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.15rem 1.25rem 0.65rem 1.25rem;
        margin-bottom: 1.15rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0B0F14;
        border-right: 1px solid #30363D;
    }

    section[data-testid="stSidebar"] * {
        color: #E6EDF3;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #B8C1CC;
    }


    /* ======================================================
       INPUT LABELS
       ====================================================== */

    label {
        color: #E6EDF3 !important;
        font-weight: 500 !important;
    }


    /* ======================================================
       NUMBER INPUTS
       ====================================================== */

    div[data-testid="stNumberInput"] input {
        background-color: #1B222C !important;
        color: #FFFFFF !important;
        border: 1px solid #3A434F !important;
        border-radius: 7px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #6E7781 !important;
        box-shadow: none !important;
    }


    /* ======================================================
       SELECTBOXES
       ====================================================== */

    div[data-baseweb="select"] > div {
        background-color: #1B222C !important;
        color: #FFFFFF !important;
        border: 1px solid #3A434F !important;
        border-radius: 7px !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] input {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] svg {
        fill: #C9D1D9 !important;
    }


    /* ======================================================
       SELECTBOX DROPDOWN MENU
       ====================================================== */

    div[data-baseweb="popover"] {
        background-color: #151A21 !important;
        border: 1px solid #30363D !important;
    }

    div[data-baseweb="menu"] {
        background-color: #151A21 !important;
    }

    div[data-baseweb="menu"] li {
        background-color: #151A21 !important;
        color: #FFFFFF !important;
    }

    div[data-baseweb="menu"] li:hover {
        background-color: #252C35 !important;
        color: #FFFFFF !important;
    }

    div[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #252C35 !important;
        color: #FFFFFF !important;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {
        min-height: 3.15rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 650;
        border: 1px solid #3A434F;
    }


    /* ======================================================
       RESULT CARD
       ====================================================== */

    .result-box {
        background-color: #151A21;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.5rem;
        min-height: 150px;
    }

    .result-label {
        color: #8B949E;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .result-condition {
        color: #FFFFFF;
        font-size: 2.25rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {
        background-color: #151A21;
        border: 1px solid #30363D;
        border-radius: 9px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #8B949E !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {
        background-color: #151A21;
        border: 1px solid #30363D;
        border-radius: 9px;
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-note {
        color: #7D8590;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 1rem;
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
        "**Track 3**  \n"
        "Climate & Environmental Health"
    )

    st.divider()

    st.markdown(
        "### About this tool"
    )

    st.write(
        "This prototype uses climate, environmental, "
        "infrastructure and population indicators to "
        "estimate an environmental health risk level."
    )

    st.write(
        "The result is intended to support environmental "
        "health screening and decision-making. It should "
        "be interpreted alongside local evidence and "
        "expert assessment."
    )

    st.divider()

    st.caption(
        "ISS AI Hackathon 2026"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Climate & Environmental Health Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted screening using climate, environmental, '
    'infrastructure and population indicators.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the available information below. "
    "The model will estimate the environmental health "
    "risk level associated with the selected profile."
)


# ============================================================
# 1. TIME & CLIMATE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '1. Time & Climate'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2100,
            value=2026
        )

    with col2:

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=8
        )

    with col3:

        temperature_c = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.5,
            step=0.1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        rainfall_mm = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=2000.0,
            value=180.0,
            step=1.0
        )

    with col2:

        relative_humidity_pct = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=78.0,
            step=1.0
        )

    with col3:

        heat_index = st.number_input(
            "Heat Index",
            min_value=-10.0,
            max_value=80.0,
            value=31.0,
            step=0.1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        rainfall_anomaly_pct = st.number_input(
            "Rainfall Anomaly (%)",
            min_value=-100.0,
            max_value=500.0,
            value=10.0,
            step=1.0
        )

    with col2:

        drought_index = st.number_input(
            "Drought Index",
            min_value=-10.0,
            max_value=10.0,
            value=0.2,
            step=0.1
        )

    with col3:

        vegetation_index = st.number_input(
            "Vegetation Index",
            min_value=-1.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )


# ============================================================
# 2. AIR QUALITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '2. Air Quality'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        pm25_ug_m3 = st.number_input(
            "PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=1000.0,
            value=35.0,
            step=1.0
        )

    with col2:

        pm10_ug_m3 = st.number_input(
            "PM10 (µg/m³)",
            min_value=0.0,
            max_value=2000.0,
            value=60.0,
            step=1.0
        )

    with col3:

        no2_ug_m3 = st.number_input(
            "NO₂ (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=1.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        o3_ug_m3 = st.number_input(
            "O₃ (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=40.0,
            step=1.0
        )

    with col2:

        co_mg_m3 = st.number_input(
            "CO (mg/m³)",
            min_value=0.0,
            max_value=100.0,
            value=0.8,
            step=0.1
        )

    with col3:

        air_quality_index = st.number_input(
            "Air Quality Index",
            min_value=0.0,
            max_value=500.0,
            value=75.0,
            step=1.0
        )


# ============================================================
# 3. FLOOD & WATER
# ============================================================

st.markdown(
    '<div class="section-title">'
    '3. Flood & Water Environment'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        flood_event = st.selectbox(
            "Flood Event",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with col2:

        flood_duration_days = st.number_input(
            "Flood Duration (days)",
            min_value=0.0,
            max_value=365.0,
            value=0.0,
            step=1.0
        )

    with col3:

        flood_prone_score = st.number_input(
            "Flood-Prone Score",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0
        )

    col1, col2 = st.columns(2)

    with col1:

        water_quality_index = st.number_input(
            "Water Quality Index",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with col2:

        sanitation_coverage_pct = st.number_input(
            "Sanitation Coverage (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )


# ============================================================
# 4. INFRASTRUCTURE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '4. Infrastructure & Sanitation'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        drainage_score = st.number_input(
            "Drainage Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    with col2:

        waste_management_score = st.number_input(
            "Waste Management Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )


# ============================================================
# 5. POPULATION & HEALTHCARE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '5. Population & Healthcare Access'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        population_density_per_km2 = st.number_input(
            "Population Density (per km²)",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            step=10.0
        )

    with col2:

        distance_to_water_km = st.number_input(
            "Distance to Water (km)",
            min_value=0.0,
            max_value=1000.0,
            value=2.0,
            step=0.1
        )

    with col3:

        healthcare_access_score = st.number_input(
            "Healthcare Access Score",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )

    vulnerable_population_pct = st.number_input(
        "Vulnerable Population (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


# ============================================================
# 6. LOCATION & SETTLEMENT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '6. Location & Settlement'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    states = [
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

    regions = [
        "North Central",
        "North East",
        "North West",
        "South East",
        "South South",
        "South West"
    ]

    urbanicity_options = [
        "Peri-urban",
        "Rural",
        "Urban"
    ]

    with col1:

        state = st.selectbox(
            "State",
            states
        )

    with col2:

        region = st.selectbox(
            "Region",
            regions
        )

    with col3:

        urbanicity = st.selectbox(
            "Urbanicity",
            urbanicity_options
        )


# ============================================================
# ASSESSMENT BUTTON
# ============================================================

st.divider()

col1, col2, col3 = st.columns(
    [1, 2, 1]
)

with col2:

    predict_button = st.button(
        "Assess Environmental Health Risk",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # RAW INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "year": year,

        "month": month,

        "temperature_c": temperature_c,

        "rainfall_mm": rainfall_mm,

        "relative_humidity_pct":
            relative_humidity_pct,

        "heat_index": heat_index,

        "pm25_ug_m3": pm25_ug_m3,

        "pm10_ug_m3": pm10_ug_m3,

        "no2_ug_m3": no2_ug_m3,

        "o3_ug_m3": o3_ug_m3,

        "co_mg_m3": co_mg_m3,

        "air_quality_index":
            air_quality_index,

        "rainfall_anomaly_pct":
            rainfall_anomaly_pct,

        "drought_index":
            drought_index,

        "vegetation_index":
            vegetation_index,

        "flood_event":
            flood_event,

        "flood_duration_days":
            flood_duration_days,

        "flood_prone_score":
            flood_prone_score,

        "water_quality_index":
            water_quality_index,

        "sanitation_coverage_pct":
            sanitation_coverage_pct,

        "drainage_score":
            drainage_score,

        "waste_management_score":
            waste_management_score,

        "population_density_per_km2":
            population_density_per_km2,

        "distance_to_water_km":
            distance_to_water_km,

        "healthcare_access_score":
            healthcare_access_score,

        "vulnerable_population_pct":
            vulnerable_population_pct,

        "state":
            state,

        "region":
            region,

        "urbanicity":
            urbanicity
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    # --------------------------------------------------------
    # EXACT RAW FEATURE ORDER
    # --------------------------------------------------------

    original_features = [

        "year",
        "month",
        "temperature_c",
        "rainfall_mm",
        "relative_humidity_pct",
        "heat_index",
        "pm25_ug_m3",
        "pm10_ug_m3",
        "no2_ug_m3",
        "o3_ug_m3",
        "co_mg_m3",
        "air_quality_index",
        "rainfall_anomaly_pct",
        "drought_index",
        "vegetation_index",
        "flood_event",
        "flood_duration_days",
        "flood_prone_score",
        "water_quality_index",
        "sanitation_coverage_pct",
        "drainage_score",
        "waste_management_score",
        "population_density_per_km2",
        "distance_to_water_km",
        "healthcare_access_score",
        "vulnerable_population_pct",
        "state",
        "region",
        "urbanicity"
    ]


    input_df = input_df[
        original_features
    ]


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = pipeline.predict(
            input_df
        )

        probabilities = pipeline.predict_proba(
            input_df
        )[0]

        predicted_class = (
            label_encoder
            .inverse_transform(
                prediction
            )[0]
        )

        max_probability = float(
            np.max(probabilities)
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Environmental Health Risk Assessment'
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
                'Estimated Risk Level'
                '</div>'
                '<div class="result-condition">'
                f'{predicted_class}'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with result_col2:

            st.metric(
                "Model Confidence",
                f"{max_probability:.1%}"
            )


        # ====================================================
        # HUMAN-READABLE INTERPRETATION
        # ====================================================

        risk_messages = {

            "Low":
                "The environmental profile is associated "
                "with a lower estimated risk level.",

            "Moderate":
                "The environmental profile is associated "
                "with a moderate estimated risk level. "
                "Further environmental assessment may "
                "be appropriate.",

            "High":
                "The environmental profile is associated "
                "with a higher estimated risk level. "
                "Further assessment and appropriate "
                "public-health action should be considered."
        }


        st.info(
            risk_messages.get(
                predicted_class,
                "Review the environmental indicators "
                "and consider further assessment."
            )
        )


        # ====================================================
        # RISK PROBABILITIES
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Risk-Level Probabilities'
            '</div>',
            unsafe_allow_html=True
        )


        probability_df = pd.DataFrame({

            "Risk Level":
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


        probability_df.insert(
            0,
            "Rank",
            range(
                1,
                len(probability_df) + 1
            )
        )


        probability_df[
            "Probability (%)"
        ] = (

            probability_df[
                "Probability"
            ] * 100

        ).round(2)


        display_df = probability_df[
            [
                "Rank",
                "Risk Level",
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
            'Relative Risk Probabilities'
            '</div>',
            unsafe_allow_html=True
        )


        chart_df = probability_df[
            [
                "Risk Level",
                "Probability (%)"
            ]
        ].set_index(
            "Risk Level"
        )


        st.bar_chart(
            chart_df
        )


        # ====================================================
        # MODEL DETAILS
        # ====================================================

        with st.expander(
            "Model information"
        ):

            st.write(
                "This prototype was developed using "
                f"{metadata.get('n_records', 'the available')} "
                "records and "
                f"{metadata.get('n_predictors', 29)} "
                "predictor variables."
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:

                st.metric(
                    "Test Accuracy",
                    f"{metadata.get('test_accuracy', 0):.1%}"
                )

            with metric_col2:

                st.metric(
                    "Balanced Accuracy",
                    f"{metadata.get('test_balanced_accuracy', 0):.1%}"
                )

            with metric_col3:

                st.metric(
                    "Macro F1",
                    f"{metadata.get('test_macro_f1', 0):.3f}"
                )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            "Important: This is an AI-assisted environmental "
            "health screening prototype developed for the "
            "ISS AI Hackathon 2026. The output represents a "
            "model estimate and should not be interpreted as "
            "a definitive prediction of health outcomes. "
            "Environmental and public-health decisions should "
            "consider additional local evidence and expert "
            "assessment."
        )


    except Exception as e:

        st.error(
            "The environmental health assessment could not "
            "be generated."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-note">'
    'ISS AI Hackathon 2026 | Track 3 | '
    'Climate & Environmental Health'
    '</div>',
    unsafe_allow_html=True
)