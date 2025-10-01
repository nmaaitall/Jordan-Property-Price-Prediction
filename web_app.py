import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(
    page_title="RealPredict",
    page_icon="🏠",
    layout="centered",  # تغيير لـ centered بدل wide
    initial_sidebar_state="collapsed"  # إخفاء Sidebar
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Poppins:wght@600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    * {
        font-family: 'Cairo', sans-serif;
    }

    .stApp {
        background: #FFFFFF;
    }

    .main .block-container {
        padding: 1.5rem 1rem;
        max-width: 500px;
    }

    /* Logo */
    .logo-text {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1rem 0 0.5rem 0;
    }

    .logo-real {
        color: #007BFF;
    }

    .logo-predict {
        color: #28A745;
    }

    .tagline {
        text-align: center;
        color: #6C757D;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Card */
    .info-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 2px solid #E9ECEF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #212529;
        margin-bottom: 1.2rem;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #007BFF 0%, #0056B3 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }

    .result-label {
        color: #FFFFFF;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    .result-price {
        color: #FFFFFF;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-currency {
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 600;
    }

    .trend-indicator {
        margin-top: 1rem;
        font-size: 0.95rem;
        font-weight: 700;
    }

    .trend-up {
        color: #28A745;
    }

    .trend-down {
        color: #DC3545;
    }

    /* Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #007BFF 0%, #0056B3 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.9rem;
        border-radius: 12px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 123, 255, 0.4);
    }

    /* Labels */
    label {
        font-weight: 700 !important;
        color: #212529 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] {
        background: #F8F9FA !important;
        border-radius: 12px !important;
        border: 2px solid #007BFF !important;
        margin-bottom: 1rem !important;
    }

    div[data-baseweb="select"] > div {
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #212529 !important;
    }

    /* Number Input */
    .stNumberInput > div > div > input {
        background: #F8F9FA !important;
        border: 2px solid #DEE2E6 !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #212529 !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #007BFF !important;
    }

    /* Slider */
    .stSlider {
        padding: 1rem 0 !important;
        margin-bottom: 1rem !important;
    }

    .stSlider > div > div > div > div {
        background: #007BFF !important;
        height: 6px !important;
    }

    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 4px solid #007BFF !important;
        width: 24px !important;
        height: 24px !important;
        box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3) !important;
    }

    /* Checkboxes */
    .stCheckbox {
        background: #F8F9FA;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        border: 2px solid #DEE2E6;
    }

    .stCheckbox label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Chart */
    .chart-container {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 2px solid #E9ECEF;
    }

    .chart-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #212529;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# تحميل النموذج
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

# Logo
st.markdown("""
<div class='logo-text'>
    <span class='logo-real'>Real</span><span class='logo-predict'>Predict</span>
</div>
<p class='tagline'>Smart Forecasts for Real Estate Prices</p>
""", unsafe_allow_html=True)

# Basic Information
st.markdown("<div class='info-card'><div class='card-title'>Basic Information</div>", unsafe_allow_html=True)

region_options = [regions_en[r] for r in regions_ar]
region = st.selectbox("Region", region_options, index=0, label_visibility="visible")
region_ar = [k for k, v in regions_en.items() if v == region][0]

area = st.number_input("Area (sqm)", 50, 1000, 150, 10)
rooms = st.number_input("Bedrooms", 1, 10, 3, 1)
bathrooms = st.number_input("Bathrooms", 1, 5, 2, 1)
age = st.number_input("Age (years)", 0, 100, 5, 1)

services = st.slider("Services Proximity (1-10)", 1, 10, 7)

st.markdown("</div>", unsafe_allow_html=True)

# Features
col1, col2 = st.columns(2)
with col1:
    elevator = st.checkbox("✓ Elevator", True)
    parking = st.checkbox("✓ Parking", True)
with col2:
    garden = st.checkbox("✓ Garden", False)
    heating = st.checkbox("✓ Heating", True)

# Button
if st.button("Start Prediction"):
    region_encoded = le.transform([region_ar])[0]

    input_data = pd.DataFrame({
        'المساحة_متر': [area],
        'عدد_الغرف': [rooms],
        'عدد_الحمامات': [bathrooms],
        'عمر_البناء_سنوات': [age],
        'طابق': [3],  # default
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

    # Result
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<div class='result-label'>Estimated Price</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-price'>{predicted_price:,.0f}</div>", unsafe_allow_html=True)
    st.markdown("<div class='result-currency'>Jordanian Dinar</div>", unsafe_allow_html=True)

    if diff_percent > 0:
        st.markdown(f"<div class='trend-indicator trend-up'>{abs(diff_percent):.1f}% above regional avg</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='trend-indicator trend-down'>{abs(diff_percent):.1f}% below regional avg</div>",
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Chart
    st.markdown("<div class='chart-container'><div class='chart-title'>Price Comparison</div>", unsafe_allow_html=True)

    fig = go.Figure(data=[
        go.Bar(
            x=['Your Price', 'Average'],
            y=[predicted_price, region_avg],
            marker_color=['#007BFF', '#6C757D'],
            text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
            textposition='outside',
            textfont=dict(size=14, weight=700)
        )
    ])

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        height=250,
        margin=dict(t=30, b=20, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor='#F1F3F5', showticklabels=False),
        xaxis=dict(showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)