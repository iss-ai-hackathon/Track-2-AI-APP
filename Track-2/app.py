import os
import json
import html
import joblib
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ISS AI Hackathon 2026 | Track 2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# DARK THEME / CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL DARK THEME
       ====================================================== */

    .stApp {
        background-color: #0B0F14;
        color: #F5F7FA;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background-color: #0F141B;
        border-right: 1px solid #252C35;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.4rem;
    }

    .sidebar-brand {
        font-size: 1.08rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.15rem;
    }

    .sidebar-event {
        font-size: 0.78rem;
        color: #A7B0BC;
        margin-bottom: 1.4rem;
    }

    .track-pill {
        display: inline-block;
        background-color: #183B63;
        color: #69B1FF;
        border: 1px solid #245B91;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .sidebar-heading {
        font-size: 0.86rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 1.25rem;
        margin-bottom: 0.45rem;
    }

    .sidebar-copy {
        color: #B8C1CC;
        font-size: 0.84rem;
        line-height: 1.65;
    }

    .sidebar-line {
        border-top: 1px solid #252C35;
        margin: 1.3rem 0;
    }

    .workflow-step {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
        color: #C2CAD3;
        font-size: 0.83rem;
        line-height: 1.45;
    }

    .step-number {
        min-width: 22px;
        height: 22px;
        border-radius: 50%;
        background-color: #18212C;
        color: #69B1FF;
        border: 1px solid #2B3948;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .eyebrow {
        color: #69B1FF;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .page-title {
        color: #FFFFFF;
        font-size: 2.4rem;
        font-weight: 750;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .page-description {
        color: #AEB7C2;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 780px;
        margin-bottom: 2rem;
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-label {
        color: #FFFFFF;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .section-help {
        color: #929CAA;
        font-size: 0.83rem;
        margin-bottom: 0.75rem;
    }


    /* ======================================================
       TEXT AREA
       ====================================================== */

    [data-testid="stTextArea"] textarea {
        background-color: #121820 !important;
        color: #FFFFFF !important;
        border: 1px solid #303945 !important;
        border-radius: 12px !important;
        font-size: 0.94rem !important;
        line-height: 1.65 !important;
        padding: 1rem !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #697482 !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #4B9CEB !important;
        box-shadow: 0 0 0 1px #4B9CEB !important;
    }


    /* ======================================================
   BUTTONS
   ====================================================== */

/* All buttons */
.stButton > button {
    border-radius: 9px !important;
    min-height: 2.75rem !important;
    font-weight: 650 !important;
}


/* Generate button */
.stButton > button[kind="primary"] {
    background-color: #1769D1 !important;
    color: #FFFFFF !important;
    border: 1px solid #1769D1 !important;
}


/* Generate button hover */
.stButton > button[kind="primary"]:hover {
    background-color: #1E7AE8 !important;
    color: #FFFFFF !important;
    border-color: #1E7AE8 !important;
}


/* Clear button */
.stButton > button[kind="secondary"] {
    background-color: #121820 !important;
    color: #FFFFFF !important;
    border: 1px solid #3A4653 !important;
}


/* Clear button hover */
.stButton > button[kind="secondary"]:hover {
    background-color: #1B232D !important;
    color: #FFFFFF !important;
    border-color: #69B1FF !important;
}


/* Button text */
.stButton > button p {
    color: #FFFFFF !important;
}

    /* ======================================================
       PREDICTION CARDS
       ====================================================== */

    .result-card {
        background-color: #121820;
        border: 1px solid #2B333D;
        border-radius: 12px;
        padding: 1.15rem 1.2rem;
        min-height: 105px;
    }

    .result-label {
        color: #8F9AA7;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.45rem;
    }

    .result-value {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 700;
    }

    .result-confidence {
        color: #69B1FF;
        font-size: 1.25rem;
        font-weight: 700;
    }


    /* ======================================================
       DOCUMENTATION CARDS
       ====================================================== */

    .doc-card {
        background-color: #121820;
        border: 1px solid #2B333D;
        border-radius: 11px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.7rem;
    }

    .doc-label {
        color: #69B1FF;
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    .doc-value {
        color: #E8ECF1;
        font-size: 0.92rem;
        line-height: 1.65;
    }


    /* ======================================================
       CONFIDENCE MESSAGES
       ====================================================== */

    .review-note {
        background-color: #241F12;
        border: 1px solid #66572C;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #E4D49A;
        font-size: 0.84rem;
        line-height: 1.55;
        margin-top: 0.8rem;
        margin-bottom: 1.3rem;
    }

    .support-note {
        background-color: #102131;
        border: 1px solid #244D70;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #A9D4F7;
        font-size: 0.84rem;
        line-height: 1.55;
        margin-top: 0.8rem;
        margin-bottom: 1.3rem;
    }


    /* ======================================================
       DISCLAIMER
       ====================================================== */

    .disclaimer {
        background-color: #151A21;
        border: 1px solid #303842;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        color: #9EA8B4;
        font-size: 0.79rem;
        line-height: 1.6;
        margin-top: 1.5rem;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #68727E;
        font-size: 0.74rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #252C35;
    }


    /* ======================================================
       STREAMLIT TEXT / LABELS
       ====================================================== */

    label {
        color: #FFFFFF !important;
    }

    [data-testid="stMarkdownContainer"] {
        color: #F5F7FA;
    }

    p {
        color: inherit;
    }

    input {
        background-color: #121820 !important;
        color: #FFFFFF !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

DIAGNOSIS_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "track2_diagnosis_model.joblib"
)

ICD10_PATH = os.path.join(
    MODEL_DIR,
    "diagnosis_to_icd10.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "track2_model_metadata.joblib"
)


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_model_package():

    diagnosis_model = joblib.load(
        DIAGNOSIS_MODEL_PATH
    )

    diagnosis_to_icd10 = joblib.load(
        ICD10_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    return (
        diagnosis_model,
        diagnosis_to_icd10,
        metadata
    )


try:

    (
        diagnosis_model,
        diagnosis_to_icd10,
        metadata
    ) = load_model_package()

except Exception as e:

    st.error(
        "The Track 2 model package could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# OPENAI CLIENT
# ============================================================

client = None

if OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# OPENAI DOCUMENTATION GENERATOR
# ============================================================

def generate_documentation(
    narrative,
    diagnosis,
    icd10
):

    prompt = f"""
You are an AI-assisted clinical documentation system.

Convert the clinical narrative below into a concise,
structured clinical documentation record.

The diagnosis classification system has already determined:

Diagnosis: {diagnosis}
ICD-10: {icd10}

These two values are fixed.

You MUST use them exactly.

Do not change the diagnosis.
Do not change the ICD-10 code.

RULES:

- Use the narrative as the primary source.
- Do not invent patient-specific findings.
- Do not fabricate examination results.
- If the narrative says there is no past medical history,
  use "None reported".
- If information is genuinely unavailable, use
  "Not documented".
- Do not add unsupported medications.
- Do not add unsupported investigations.
- Keep the documentation concise.
- Do not introduce additional diagnoses.

Generate a SOAP note:

S: Subjective information
O: Objective information
A: Assessment
P: Plan

Return ONLY valid JSON.

Use exactly these fields:

{{
    "chief_complaint": "",
    "hpi": "",
    "pmh": "",
    "exam": "",
    "differential": "",
    "final_diagnosis": "",
    "icd10": "",
    "investigations": "",
    "medications": "",
    "treatment_plan": "",
    "soap_note": ""
}}

Clinical narrative:

{narrative}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return json.loads(
        response.output_text
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            ISS AI Hackathon 2026
        </div>

        <div class="sidebar-event">
            Artificial Intelligence for Health
        </div>

        <div class="track-pill">
            TRACK 2
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-heading">
            What is this?
        </div>

        <div class="sidebar-copy">
            A clinical documentation assistant that turns
            free-text clinical narratives into structured
            documentation and coding support.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-heading">
            How it works
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="workflow-step">
            <div class="step-number">1</div>
            <div>Enter a patient's clinical narrative.</div>
        </div>

        <div class="workflow-step">
            <div class="step-number">2</div>
            <div>
                AI classifies the most likely standardized
                diagnosis.
            </div>
        </div>

        <div class="workflow-step">
            <div class="step-number">3</div>
            <div>
                The corresponding ICD-10 code is retrieved.
            </div>
        </div>

        <div class="workflow-step">
            <div class="step-number">4</div>
            <div>
                AI generates structured documentation and
                a SOAP note.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-heading">
            Built for the hackathon
        </div>

        <div class="sidebar-copy">
            The system combines a trained clinical text
            classification model with an AI documentation
            layer to support faster and more consistent
            clinical documentation.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-heading">
            Important
        </div>

        <div class="sidebar-copy">
            This is a decision-support prototype.
            It does not replace clinical assessment,
            professional judgment, or patient care.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">'
    'ISS AI HACKATHON 2026 · TRACK 2'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-title">'
    'Clinical Documentation Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-description">'
    'Turn an unstructured clinical narrative into a clear, '
    'structured clinical record with AI-assisted classification, '
    'ICD-10 coding support, and SOAP documentation.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CLINICAL NARRATIVE INPUT
# ============================================================

st.markdown(
    '<div class="section-label">'
    'What did the patient present with?'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-help">'
    'Paste the clinical narrative below. The more complete '
    'the narrative, the more useful the generated documentation.'
    '</div>',
    unsafe_allow_html=True
)


narrative = st.text_area(
    "Clinical narrative",
    height=210,
    placeholder=(
        "Example:\n\n"
        "Patient presents with fever, headache and generalized "
        "body pains for three days. Past medical history: none "
        "reported. Vitals stable."
    ),
    label_visibility="collapsed"
)


# ============================================================
# ACTION BUTTONS
# ============================================================

button_col1, button_col2 = st.columns([4, 1])


with button_col1:

    generate_button = st.button(
        "Generate documentation",
        type="primary",
        use_container_width=True
    )


with button_col2:

    clear_button = st.button(
        "Clear",
        use_container_width=True
    )


if clear_button:

    st.rerun()


# ============================================================
# PROCESS NARRATIVE
# ============================================================

if generate_button:

    if not narrative.strip():

        st.warning(
            "Enter a clinical narrative before generating documentation."
        )

        st.stop()


    # ========================================================
    # DIAGNOSIS PREDICTION
    # ========================================================

    prediction = diagnosis_model.predict(
        [narrative]
    )[0]


    # ========================================================
    # MODEL CONFIDENCE
    # ========================================================

    probabilities = diagnosis_model.predict_proba(
        [narrative]
    )[0]

    confidence = float(
        probabilities.max()
    )


    # ========================================================
    # ICD-10 LOOKUP
    # ========================================================

    icd10 = diagnosis_to_icd10.get(
        prediction,
        "Not available"
    )


    # ========================================================
    # AI CLASSIFICATION RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'AI-assisted classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-help">'
        'The model identifies the most likely standardized '
        'diagnosis from the narrative.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Standardized diagnosis
                </div>

                <div class="result-value">
                    {html.escape(str(prediction))}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    ICD-10
                </div>

                <div class="result-value">
                    {html.escape(str(icd10))}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Model confidence
                </div>

                <div class="result-confidence">
                    {confidence:.2%}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # CONFIDENCE INTERPRETATION
    # ========================================================

    if confidence < 0.50:

        st.markdown(
            """
            <div class="review-note">
                <strong>Low confidence.</strong>
                The model is uncertain about this classification.
                Review the narrative and verify the result clinically.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif confidence < 0.70:

        st.markdown(
            """
            <div class="review-note">
                <strong>Moderate confidence.</strong>
                The prediction should be reviewed by a qualified
                healthcare professional.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="support-note">
                <strong>Higher model confidence.</strong>
                The classification is relatively confident, but
                professional clinical review is still required.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # STRUCTURED DOCUMENTATION
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'Structured clinical documentation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-help">'
        'The AI documentation layer organizes the narrative '
        'into a standardized clinical record.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # CHECK OPENAI
    # ========================================================

    if client is None:

        st.error(
            "OpenAI API key was not found. "
            "Check your .env file."
        )

        st.stop()


    # ========================================================
    # GENERATE DOCUMENTATION
    # ========================================================

    with st.spinner(
        "Preparing clinical documentation..."
    ):

        try:

            documentation = generate_documentation(
                narrative=narrative,
                diagnosis=prediction,
                icd10=icd10
            )

        except Exception as e:

            st.error(
                "The documentation service could not complete the request."
            )

            st.code(str(e))

            st.stop()


    # ========================================================
    # DISPLAY DOCUMENTATION
    # ========================================================

    fields = [
        ("Chief Complaint", "chief_complaint"),
        ("History of Present Illness", "hpi"),
        ("Past Medical History", "pmh"),
        ("Examination", "exam"),
        ("Differential", "differential"),
        ("Final Diagnosis", "final_diagnosis"),
        ("ICD-10", "icd10"),
        ("Investigations", "investigations"),
        ("Medications", "medications"),
        ("Treatment Plan", "treatment_plan"),
        ("SOAP Note", "soap_note")
    ]


    for label, key in fields:

        value = documentation.get(
            key,
            "Not documented"
        )

        value = html.escape(
            str(value)
        )

        st.markdown(
            f"""
            <div class="doc-card">

                <div class="doc-label">
                    {label}
                </div>

                <div class="doc-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # CLINICAL DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="disclaimer">

        <strong>Clinical decision-support notice:</strong>
        This application is a hackathon prototype for AI-assisted
        clinical documentation and classification. Generated
        information may contain errors and must be reviewed and
        verified by a qualified healthcare professional before
        clinical use.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ISS AI Hackathon 2026 · Track 2 · Clinical Documentation Assistant
    </div>
    """,
    unsafe_allow_html=True
)