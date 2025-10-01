import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(
    page_title="RealPredict - Smart Real Estate Valuation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px rgba(30, 60, 114, 0.25);
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .logo-container {
        position: relative;
        z-index: 1;
    }

    .logo-text {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
        letter-spacing: -1px;
    }

    .logo-icon {
        display: inline-block;
        margin-right: 0.5rem;
        font-size: 3rem;
    }

    .tagline {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.15rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .subtitle {
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.95rem;
        margin-top: 1rem;
        font-weight: 400;
    }

    /* Main Grid Layout */
    .input-section {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8ecf1;
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #2a5298;
        display: flex;
        align-items: center;
    }

    .section-icon {
        margin-right: 0.75rem;
        font-size: 1.6rem;
    }

    /* Enhanced Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(30, 60, 114, 0.35);
        position: relative;
        overflow: hidden;
        min-height: 380px;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }

    .result-content {
        position: relative;
        z-index: 1;
    }

    .result-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    .result-price {
        color: #ffffff;
        font-size: 4.5rem;
        font-weight: 900;
        margin: 1.5rem 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        line-height: 1;
        letter-spacing: -2px;
    }

    .result-currency {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }

    .trend-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1.25rem 2rem;
        margin: 2rem auto 0 auto;
        max-width: 85%;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }

    .trend-content {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }

    .trend-icon {
        font-size: 2rem;
        font-weight: 800;
    }

    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }

    .trend-text {
        color: #1e293b;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }

    /* Enhanced Form Elements */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }

    /* Selectbox Styling */
    div[data-baseweb="select"] {
        background: #f8fafc !important;
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }

    div[data-baseweb="select"]:hover {
        border-color: #2a5298 !important;
    }

    div[data-baseweb="select"] > div {
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        background: transparent !important;
    }

    div[data-baseweb="popover"] {
        background: #ffffff !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15) !important;
    }

    ul[role="listbox"] {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
    }

    ul[role="listbox"] li {
        color: #1e293b !important;
        background: #ffffff !important;
        padding: 1rem 1.25rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    ul[role="listbox"] li:hover {
        background: #f1f5f9 !important;
        color: #2a5298 !important;
    }

    ul[role="listbox"] li[aria-selected="true"] {
        background: #2a5298 !important;
        color: #ffffff !important;
    }

    /* Number Input */
    .stNumberInput > div > div > input {
        background: #f8fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        transition: all 0.3s ease !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #2a5298 !important;
        box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1) !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%) !important;
        height: 6px !important;
    }

    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 4px solid #2a5298 !important;
        width: 26px !important;
        height: 26px !important;
        box-shadow: 0 3px 12px rgba(42, 82, 152, 0.4) !important;
    }

    /* Checkboxes */
    .stCheckbox {
        background: #f8fafc;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .stCheckbox:hover {
        border-color: #cbd5e1;
        background: #f1f5f9;
    }

    .stCheckbox label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #334155 !important;
    }

    /* Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 1.15rem;
        border-radius: 14px;
        border: none;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.35);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(42, 82, 152, 0.45);
    }

    /* Chart Container */
    .chart-section {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8ecf1;
    }

    .chart-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 1.5rem;
    }

    /* Info Cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .info-card-mini {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border: 2px solid #e2e8f0;
        text-align: center;
    }

    .info-card-mini .label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .info-card-mini .value {
        font-size: 1.5rem;
        color: #1e3c72;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model():
    df = pd.read_csv('jordan_properties.csv')
    le = LabelEncoder()
    df['المنطقة_رقم'] = le.fit_transform(df['المنطقة'])

    features = ['المساحة_متر', 'عدد_الغرف', 'عدد_الحمامات', 'عمر_البناء_سنوات',
                'طابق', 'يوجد_مصعد', 'يوجد_موقف', 'يوجد_حديقة',
                'يوجد_تدفئة_مركزية', 'قرب_الخدمات', 'المنطقة_رقم']

    X = df[features]
    y = df['السعر_دينار']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    regions_ar = sorted(df['المنطقة'].unique())
    regions_en = {
        'عبدون': 'Abdoun', 'دير غبار': 'Deir Ghbar', 'أم أذينة': 'Um Uthaina',
        'الصويفية': 'Sweifieh', 'خلدا': 'Khalda', 'أم السماق': 'Um Summaq',
        'تلاع العلي': 'Tla Al Ali', 'الجاردنز': 'Gardens', 'الشميساني': 'Shmeisani',
        'اللويبدة': 'Luweibdeh', 'الجبيهة': 'Jubeiha', 'صويلح': 'Sweileh',
        'طبربور': 'Tabarbour', 'ماركا': 'Marka', 'شفا بدران': 'Shafa Badran',
        'الياسمين': 'Yasmin', 'المقابلين': 'Maqablain', 'الهاشمي الشمالي': 'Hashemi North',
        'جبل الحسين': 'Jabal Hussein', 'النصر': 'Nasr'
    }

    return model, le, regions_ar, regions_en, df


model, le, regions_ar, regions_en, df = load_model()

# Header
st.markdown("""
<div class='header-container'>
    <div class='logo-container'>
        <div class='logo-text'>
            <span class='logo-icon'>🏢</span>RealPredict
        </div>
        <div class='tagline'>Professional Real Estate Valuation Platform</div>
        <div class='subtitle'>Powered by Advanced Machine Learning & Market Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Layout
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    # Location & Basic Info
    st.markdown(
        "<div class='input-section'><div class='section-header'><span class='section-icon'>📍</span>Property Location & Details</div>",
        unsafe_allow_html=True)

    region_options = [regions_en[r] for r in regions_ar]
    region = st.selectbox("Select Region", region_options, index=0)
    region_ar = [k for k, v in regions_en.items() if v == region][0]

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (Square Meters)", 50, 1000, 150, 10)
        rooms = st.number_input("Number of Bedrooms", 1, 10, 3, 1)
    with col2:
        bathrooms = st.number_input("Number of Bathrooms", 1, 5, 2, 1)
        age = st.number_input("Property Age (Years)", 0, 100, 5, 1)

    st.markdown("</div>", unsafe_allow_html=True)

    # Amenities
    st.markdown(
        "<div class='input-section'><div class='section-header'><span class='section-icon'>✨</span>Property Features & Amenities</div>",
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        elevator = st.checkbox("🛗 Elevator Available", True)
        parking = st.checkbox("🚗 Parking Space", True)
    with col2:
        garden = st.checkbox("🌳 Private Garden", False)
        heating = st.checkbox("🔥 Central Heating", True)

    st.markdown("<br>", unsafe_allow_html=True)
    services = st.slider("Proximity to Services & Amenities (1-10)", 1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Button
    predict_button = st.button("🔍 Calculate Property Value")

with col_right:
    if predict_button:
        region_encoded = le.transform([region_ar])[0]

        input_data = pd.DataFrame({
            'المساحة_متر': [area],
            'عدد_الغرف': [rooms],
            'عدد_الحمامات': [bathrooms],
            'عمر_البناء_سنوات': [age],
            'طابق': [3],
            'يوجد_مصعد': [1 if elevator else 0],
            'يوجد_موقف': [1 if parking else 0],
            'يوجد_حديقة': [1 if garden else 0],
            'يوجد_تدفئة_مركزية': [1 if heating else 0],
            'قرب_الخدمات': [services],
            'المنطقة_رقم': [region_encoded]
        })

        predicted_price = model.predict(input_data)[0]
        region_avg = df[df['المنطقة'] == region_ar]['السعر_دينار'].mean()
        diff_percent = ((predicted_price - region_avg) / region_avg) * 100

        # Result Card
        if diff_percent > 0:
            trend_html = f"""
                <div class='trend-container'>
                    <div class='trend-content'>
                        <span class='trend-icon trend-up'>↑</span>
                        <p class='trend-text'>{abs(diff_percent):.1f}% Above Regional Average</p>
                    </div>
                </div>
            """
        else:
            trend_html = f"""
                <div class='trend-container'>
                    <div class='trend-content'>
                        <span class='trend-icon trend-down'>↓</span>
                        <p class='trend-text'>{abs(diff_percent):.1f}% Below Regional Average</p>
                    </div>
                </div>
            """

        st.markdown(f"""
            <div class='result-card'>
                <div class='result-content'>
                    <div class='result-badge'>Estimated Market Value</div>
                    <div class='result-price'>{predicted_price:,.0f}</div>
                    <div class='result-currency'>Jordanian Dinar (JOD)</div>
                    {trend_html}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Info Cards
        st.markdown("""
        <div class='info-grid'>
            <div class='info-card-mini'>
                <div class='label'>Price/SQM</div>
                <div class='value'>""" + f"{predicted_price / area:,.0f}" + """</div>
            </div>
            <div class='info-card-mini'>
                <div class='label'>Region Avg</div>
                <div class='value'>""" + f"{region_avg:,.0f}" + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chart
        st.markdown("<div class='chart-section'><div class='chart-title'>Market Comparison Analysis</div>",
                    unsafe_allow_html=True)

        fig = go.Figure(data=[
            go.Bar(
                x=['Your Property', 'Regional Average'],
                y=[predicted_price, region_avg],
                marker=dict(
                    color=['#2a5298', '#94a3b8'],
                    line=dict(color='#1e3c72', width=2)
                ),
                text=[f'{predicted_price:,.0f} JOD', f'{region_avg:,.0f} JOD'],
                textposition='outside',
                textfont=dict(size=13, weight=700, color='#1e3c72')
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=280,
            margin=dict(t=40, b=30, l=20, r=20),
            yaxis=dict(
                showgrid=True,
                gridcolor='#e2e8f0',
                showticklabels=True,
                tickfont=dict(size=11, color='#64748b')
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=12, weight=600, color='#1e3c72')
            ),
            font=dict(family='Inter, sans-serif')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='result-card'>
                <div class='result-content'>
                    <div class='result-badge'>Ready to Estimate</div>
                    <div style='font-size: 4rem; margin: 2rem 0;'>🏢</div>
                    <div class='result-currency' style='margin-bottom: 1rem;'>Enter property details and click</div>
                    <div class='result-currency'>"Calculate Property Value"</div>
                </div>
            </div>
        """, unsafe_allow_html=True)