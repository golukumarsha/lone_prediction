import streamlit as st
import numpy as np
import joblib

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ · Smart Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark Gold Banking Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e0cc;
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Background texture ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(180,145,60,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(180,145,60,0.05) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 60px, rgba(180,145,60,0.015) 60px, rgba(180,145,60,0.015) 61px),
        repeating-linear-gradient(90deg, transparent, transparent 60px, rgba(180,145,60,0.015) 60px, rgba(180,145,60,0.015) 61px);
    pointer-events: none;
    z-index: 0;
}

/* ── Hero Header ── */
.hero-wrap {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 52px 24px 36px;
    margin-bottom: 8px;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 4px;
    color: #b4913c;
    text-transform: uppercase;
    margin-bottom: 14px;
    opacity: 0.9;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(40px, 6vw, 68px);
    font-weight: 700;
    color: #f5e6c0;
    line-height: 1;
    margin-bottom: 16px;
    letter-spacing: -1px;
}
.hero-title span {
    background: linear-gradient(135deg, #d4a843, #f0cd7a, #b4913c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 15px;
    color: #7a7060;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
}
.hero-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    max-width: 320px;
    margin: 28px auto 0;
}
.hero-divider-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(180,145,60,0.4), transparent);
}
.hero-divider-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #b4913c;
    opacity: 0.6;
}

/* ── Section Header ── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 18px;
}
.section-hdr-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: rgba(180,145,60,0.12);
    border: 1px solid rgba(180,145,60,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.section-hdr-text {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #b4913c;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(180,145,60,0.12);
    border-radius: 18px;
    padding: 24px 26px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,145,60,0.3), transparent);
}

/* ── Labels ── */
label, .stNumberInput label, .stSelectbox label,
div[data-testid="stFormLabel"], .stSlider label {
    color: #9a8a6a !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
    margin-bottom: 6px !important;
}

/* ── Number Inputs ── */
input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(180,145,60,0.2) !important;
    border-radius: 10px !important;
    color: #f5e6c0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s !important;
}
input[type="number"]:focus {
    border-color: rgba(180,145,60,0.6) !important;
    box-shadow: 0 0 0 3px rgba(180,145,60,0.08) !important;
    outline: none !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(180,145,60,0.2) !important;
    border-radius: 10px !important;
    color: #f5e6c0 !important;
    font-size: 15px !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(180,145,60,0.5) !important;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #b4913c, #d4a843, #b4913c) !important;
    background-size: 200% !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 48px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    font-family: 'DM Mono', monospace !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(180,145,60,0.3), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(180,145,60,0.5) !important;
    background-position: right !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result Box ── */
.result-approved {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(34,197,94,0.04));
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-approved::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #22c55e, transparent);
}
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-rejected::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ef4444, transparent);
}
.result-icon { font-size: 52px; margin-bottom: 14px; line-height: 1; }
.result-status {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 10px;
}
.result-status-approved { color: #4ade80; }
.result-status-rejected { color: #f87171; }
.result-msg {
    font-size: 13px;
    color: #7a7060;
    letter-spacing: 0.5px;
    line-height: 1.5;
}

/* ── Summary Strip ── */
.summary-strip {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 20px 0;
}
.summary-chip {
    background: rgba(180,145,60,0.08);
    border: 1px solid rgba(180,145,60,0.15);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 11px;
    color: #9a8a6a;
    font-family: 'DM Mono', monospace;
}
.summary-chip span {
    color: #d4a843;
    font-weight: 500;
    margin-left: 4px;
}

/* ── Info Sidebar panel ── */
.info-panel {
    background: rgba(180,145,60,0.05);
    border: 1px solid rgba(180,145,60,0.12);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.info-panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #b4913c;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-weight: 500;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(180,145,60,0.07);
    font-size: 12px;
}
.info-row:last-child { border-bottom: none; }
.info-row-label { color: #6a5a4a; }
.info-row-val { color: #c4a85c; font-family: 'DM Mono', monospace; font-size: 11px; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 32px 16px 24px;
    font-size: 11px;
    color: #3a3028;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
}
.footer-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(180,145,60,0.15), transparent);
    margin-bottom: 20px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 12, 0.97) !important;
    border-right: 1px solid rgba(180,145,60,0.1) !important;
}
[data-testid="stSidebar"] * { color: #9a8a6a !important; }

/* ── Column gap fix ── */
[data-testid="column"] { padding: 0 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    model_err = str(e)

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">AI-Powered · Instant Decision</div>
    <h1 class="hero-title">Loan<span>IQ</span></h1>
    <p class="hero-sub">Enter your financial profile and receive an intelligent loan eligibility assessment in seconds.</p>
    <div class="hero-divider">
        <div class="hero-divider-line"></div>
        <div class="hero-divider-dot"></div>
        <div class="hero-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(
        f"⚠️ Model files not found! Place `logistic_regression_model.pkl` and `scaler.pkl` in the same folder.\n\n**Error:** {model_err}")
    st.stop()

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:

    # ── Section 1: Personal Info ──
    st.markdown("""
    <div class="section-hdr">
        <div class="section-hdr-icon">👤</div>
        <div class="section-hdr-text">Personal Information</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        Married = st.selectbox("Marital Status", ["Yes", "No"])
    with c3:
        Dependents = st.selectbox("Dependents", [0, 1, 2, 3])

    c4, c5 = st.columns(2)
    with c4:
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c5:
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Financial Info ──
    st.markdown("""
    <div class="section-hdr">
        <div class="section-hdr-icon">💰</div>
        <div class="section-hdr-text">Financial Details</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        ApplicantIncome = st.number_input(
            "Applicant Income (₹)", min_value=0, value=50000, step=1000)
    with f2:
        CoapplicantIncome = st.number_input(
            "Co-applicant Income (₹)", min_value=0, value=0, step=1000)

    f3, f4 = st.columns(2)
    with f3:
        LoanAmount = st.number_input(
            "Loan Amount (₹ thousands)", min_value=0, value=150, step=10)
    with f4:
        Loan_Amount_Term = st.number_input(
            "Loan Term (months)", min_value=0, value=360, step=12)

    Credit_History = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x: "✅ Good (Cleared debts)" if x == 1 else "❌ Poor (Pending debts)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Property ──
    st.markdown("""
    <div class="section-hdr">
        <div class="section-hdr-icon">🏘️</div>
        <div class="section-hdr-text">Property Details</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    Property_Area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"],
        help="Location type of the property to be purchased"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Check Loan Eligibility")


# ─────────────────────────────────────────────
#  RIGHT COLUMN — Summary + Result
# ─────────────────────────────────────────────
with right_col:

    # Live Summary
    Total_Income = ApplicantIncome + CoapplicantIncome
    debt_ratio = round((LoanAmount / (Total_Income / 1000))
                       * 100, 1) if Total_Income > 0 else 0

    st.markdown("""
    <div class="section-hdr">
        <div class="section-hdr-icon">📊</div>
        <div class="section-hdr-text">Application Summary</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-panel">
        <div class="info-panel-title">Financial Overview</div>
        <div class="info-row">
            <span class="info-row-label">Total Monthly Income</span>
            <span class="info-row-val">₹{Total_Income:,}</span>
        </div>
        <div class="info-row">
            <span class="info-row-label">Loan Amount</span>
            <span class="info-row-val">₹{LoanAmount}K</span>
        </div>
        <div class="info-row">
            <span class="info-row-label">Loan Tenure</span>
            <span class="info-row-val">{Loan_Amount_Term} months</span>
        </div>
        <div class="info-row">
            <span class="info-row-label">Debt Ratio</span>
            <span class="info-row-val">{debt_ratio}%</span>
        </div>
        <div class="info-row">
            <span class="info-row-label">Credit History</span>
            <span class="info-row-val">{'Good ✓' if Credit_History == 1 else 'Poor ✗'}</span>
        </div>
        <div class="info-row">
            <span class="info-row-label">Dependents</span>
            <span class="info-row-val">{Dependents}</span>
        </div>
        <div class="info-row">
            <span class="info-row-label">Property Area</span>
            <span class="info-row-val">{Property_Area}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Profile chips
    st.markdown(f"""
    <div class="summary-strip">
        <div class="summary-chip">{Gender}</div>
        <div class="summary-chip">{'Married' if Married == 'Yes' else 'Single'}</div>
        <div class="summary-chip">{Education}</div>
        <div class="summary-chip">{'Self-Employed' if Self_Employed == 'Yes' else 'Salaried'}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Result ──
    if predict_btn:

        # Encoding
        Gender_Female = 1 if Gender == "Female" else 0
        Gender_Male = 1 if Gender == "Male" else 0
        Married_Yes = 1 if Married == "Yes" else 0
        Married_No = 1 if Married == "No" else 0
        Education_Graduate = 1 if Education == "Graduate" else 0
        Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
        Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
        Self_Employed_No = 1 if Self_Employed == "No" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Rural = 1 if Property_Area == "Rural" else 0

        input_data = np.array([[
            Dependents, ApplicantIncome, CoapplicantIncome,
            LoanAmount, Loan_Amount_Term, Credit_History, Total_Income,
            Gender_Female, Gender_Male,
            Married_No, Married_Yes,
            Education_Graduate, Education_Not_Graduate,
            Self_Employed_No, Self_Employed_Yes,
            Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
        ]])

        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.markdown("""
                <div class="result-approved">
                    <div class="result-icon">✅</div>
                    <div class="result-status result-status-approved">Approved</div>
                    <div class="result-msg">Congratulations! Based on your financial<br>profile, your loan application is eligible.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-rejected">
                    <div class="result-icon">❌</div>
                    <div class="result-status result-status-rejected">Not Approved</div>
                    <div class="result-msg">Your current profile does not meet the<br>eligibility criteria. Consider improving<br>your credit score or increasing income.</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="background:rgba(180,145,60,0.04); border:1px dashed rgba(180,145,60,0.15);
                    border-radius:20px; padding:48px 32px; text-align:center; margin-top:8px;">
            <div style="font-size:40px; margin-bottom:14px; opacity:0.4;">🏦</div>
            <div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px;
                        color:#4a3a2a; text-transform:uppercase;">Awaiting Assessment</div>
            <div style="font-size:13px; color:#3a2a1a; margin-top:8px; line-height:1.5;">
                Fill in your details and click<br><em>Check Loan Eligibility</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-divider"></div>
    LoanIQ · Powered by Logistic Regression · For educational purposes only
</div>
""", unsafe_allow_html=True)
