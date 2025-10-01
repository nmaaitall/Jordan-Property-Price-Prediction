import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# إعدادات الصفحة
st.set_page_config(
    page_title="Jordan Property Price Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    * {
        font-family: 'Cairo', 'Segoe UI', sans-serif;
    }

    /* خلفية بيضاء */
    .stApp {
        background: #FFFFFF;
    }

    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1500px;
    }

    /* العنوان الرئيسي - نص أسود */
    .main-title {
        text-align: center;
        color: #212529;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #495057;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* البطاقات - رمادي فاتح مع ظل */
    .custom-card {
        background: #F8F9FA;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #E9ECEF;
    }

    /* عناوين البطاقات - أسود غامق */
    .card-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #212529;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #007BFF;
    }

    /* بطاقة النتيجة - أزرق */
    .result-card {
        background: linear-gradient(135deg, #007BFF 0%, #0056B3 100%);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.3);
        margin-top: 1.5rem;
    }

    .result-text {
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }

    .result-price {
        color: #FFFFFF;
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-currency {
        color: #FFFFFF;
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.9;
    }

    /* الأزرار - أزرق */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #007BFF 0%, #0056B3 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
        transition: all 0.3s ease;
        margin-top: 1.5rem;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4);
        background: linear-gradient(135deg, #0056B3 0%, #004085 100%);
    }

    /* Labels - أسود غامق */
    label {
        font-weight: 700 !important;
        color: #212529 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Selectbox - خلفية بيضاء */
    div[data-baseweb="select"] {
        background: #FFFFFF !important;
        border-radius: 12px !important;
        border: 2px solid #DEE2E6 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }

    div[data-baseweb="select"]:hover {
        border-color: #007BFF !important;
    }

    div[data-baseweb="select"] > div {
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        color: #212529 !important;
        font-weight: 600 !important;
    }

    /* Number Input - خلفية بيضاء */
    .stNumberInput > div > div > input {
        background: #FFFFFF !important;
        border: 2px solid #DEE2E6 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        color: #212529 !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #007BFF !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1) !important;
    }

    /* أزرار Number Input */
    .stNumberInput button {
        background: #F1F3F5 !important;
        color: #495057 !important;
        border: 1px solid #DEE2E6 !important;
        border-radius: 8px !important;
    }

    .stNumberInput button:hover {
        background: #007BFF !important;
        color: white !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: #007BFF !important;
    }

    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid #007BFF !important;
        box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3) !important;
    }

    /* Checkboxes - خلفية بيضاء */
    .stCheckbox {
        background: #FFFFFF;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 2px solid #DEE2E6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }

    .stCheckbox:hover {
        border-color: #007BFF;
        transform: translateX(-2px);
    }

    .stCheckbox label {
        font-weight: 600 !important;
        color: #212529 !important;
        font-size: 0.95rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #F8F9FA;
        border-right: 1px solid #DEE2E6;
    }

    /* Info Box */
    .info-box {
        background: #E7F3FF;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #007BFF;
        margin: 1rem 0;
        color: #212529;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .info-box strong {
        color: #0056B3;
        display: block;
        margin-bottom: 0.3rem;
        font-size: 1rem;
    }

    /* Stats Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #212529 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #495057 !important;
        font-weight: 600 !important;
    }

    /* Initial State */
    .initial-state {
        background: #F8F9FA;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        border: 2px dashed #CED4DA;
        margin-top: 1.5rem;
    }

    .initial-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }

    .initial-state-title {
        font-size: 1.4rem;
        color: #212529;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .initial-state-desc {
        color: #6C757D;
        font-size: 1rem;
    }

    /* Radio Buttons */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #DEE2E6;
    }

    /* Trend indicators */
    .trend-up {
        color: #28A745;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    .trend-down {
        color: #DC3545;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #DEE2E6;
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

# Sidebar
with st.sidebar:
    st.markdown("### اللغة / Language")
    language = st.radio("", ["العربية", "English"], label_visibility="collapsed")
    is_arabic = language == "العربية"

    st.markdown("<hr>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("### إحصائيات النظام")
    else:
        st.markdown("### System Statistics")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("العقارات" if is_arabic else "Properties", f"{len(df):,}")
    with col_s2:
        st.metric("المناطق" if is_arabic else "Regions", "20")

    avg_price = df['السعر_دينار'].mean()
    st.metric(
        "متوسط السعر" if is_arabic else "Avg Price",
        f"{avg_price:,.0f} د.أ" if is_arabic else f"{avg_price:,.0f} JOD"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("""
        <div class='info-box'>
        <strong>كيف يعمل النظام؟</strong>
        يستخدم خوارزميات التعلم الآلي المدربة على 1,500 عقار من 20 منطقة مختلفة لتوقع الأسعار بدقة عالية
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
        <strong>How It Works?</strong>
        Uses ML algorithms trained on 1,500 properties from 20 regions to predict prices with high accuracy
        </div>
        """, unsafe_allow_html=True)

# العنوان
if is_arabic:
    st.markdown("<h1 class='main-title'>نظام التنبؤ بأسعار العقارات في الأردن</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>تقييم دقيق وذكي باستخدام الذكاء الاصطناعي</p>", unsafe_allow_html=True)
else:
    st.markdown("<h1 class='main-title'>Jordan Property Price Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Accurate & Smart Valuation Using AI</p>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1.4, 1], gap="large")

with col1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("<div class='card-header'>المعلومات الأساسية</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>Basic Information</div>", unsafe_allow_html=True)

    # المنطقة
    if is_arabic:
        region_options = regions_ar
        region = st.selectbox("المنطقة", region_options, index=0)
        region_ar = region
    else:
        region_options = [regions_en[r] for r in regions_ar]
        region = st.selectbox("Region", region_options, index=0)
        region_ar = [k for k, v in regions_en.items() if v == region][0]

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        area = st.number_input("المساحة (م²)" if is_arabic else "Area (sqm)", 50, 1000, 150, 10)
    with col_b:
        rooms = st.number_input("غرف النوم" if is_arabic else "Bedrooms", 1, 10, 3, 1)

    col_c, col_d = st.columns(2, gap="medium")
    with col_c:
        bathrooms = st.number_input("الحمامات" if is_arabic else "Bathrooms", 1, 5, 2, 1)
    with col_d:
        age = st.number_input("عمر البناء" if is_arabic else "Age (years)", 0, 100, 5, 1)

    col_e, col_f = st.columns(2, gap="medium")
    with col_e:
        floor = st.number_input("الطابق" if is_arabic else "Floor", 0, 20, 3, 1)

    services = st.slider("قرب الخدمات (1-10)" if is_arabic else "Services Proximity (1-10)", 1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # المميزات
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    if is_arabic:
        st.markdown("<div class='card-header'>المميزات والخدمات</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>Features & Amenities</div>", unsafe_allow_html=True)

    col_g, col_h = st.columns(2, gap="medium")
    with col_g:
        elevator = st.checkbox("يوجد مصعد" if is_arabic else "Elevator", True)
        garden = st.checkbox("يوجد حديقة" if is_arabic else "Garden", False)
    with col_h:
        parking = st.checkbox("يوجد موقف" if is_arabic else "Parking", True)
        heating = st.checkbox("تدفئة مركزية" if is_arabic else "Heating", True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    calculate_btn = st.button("احسب السعر المتوقع" if is_arabic else "Calculate Price", use_container_width=True)

    if calculate_btn:
        region_encoded = le.transform([region_ar])[0]

        input_data = pd.DataFrame({
            'المساحة_متر': [area],
            'عدد_الغرف': [rooms],
            'عدد_الحمامات': [bathrooms],
            'عمر_البناء_سنوات': [age],
            'طابق': [floor],
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

        # عرض النتيجة
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        if is_arabic:
            st.markdown("<div class='result-text'>السعر المتوقع</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} د.أ</div>", unsafe_allow_html=True)

            if diff_percent > 0:
                st.markdown(f"<div class='trend-up'>↑ أعلى بـ {abs(diff_percent):.1f}% من متوسط المنطقة</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='trend-down'>↓ أقل بـ {abs(diff_percent):.1f}% من متوسط المنطقة</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-text'>Estimated Price</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} JOD</div>", unsafe_allow_html=True)

            if diff_percent > 0:
                st.markdown(f"<div class='trend-up'>↑ {abs(diff_percent):.1f}% above regional avg</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='trend-down'>↓ {abs(diff_percent):.1f}% below regional avg</div>",
                            unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # رسم بياني
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        if is_arabic:
            st.markdown("<div class='card-header'>مقارنة الأسعار</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-header'>Price Comparison</div>", unsafe_allow_html=True)

        fig = go.Figure(data=[
            go.Bar(
                x=[('سعرك' if is_arabic else 'Your Price'),
                   ('المتوسط' if is_arabic else 'Average')],
                y=[predicted_price, region_avg],
                marker_color=['#007BFF', '#6C757D'],
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=14, color='#212529', weight=700)
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=300,
            margin=dict(t=40, b=30, l=20, r=20),
            yaxis=dict(showgrid=True, gridcolor='#E9ECEF', showticklabels=True,
                       tickfont=dict(size=11, color='#495057')),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color='#212529', weight=600))
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='initial-state'>", unsafe_allow_html=True)
        st.markdown("<div class='initial-state-icon'>🏛️</div>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='initial-state-title'>ابدأ التقييم</div>", unsafe_allow_html=True)
            st.markdown("<div class='initial-state-desc'>أدخل معلومات العقار واضغط زر الحساب</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='initial-state-title'>Start Valuation</div>", unsafe_allow_html=True)
            st.markdown("<div class='initial-state-desc'>Enter property details and calculate</div>",
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #6C757D; font-size: 0.9rem; padding: 1.5rem 0; margin-top: 2rem; border-top: 2px solid #DEE2E6;'>
    <div style='margin-bottom: 0.3rem;'>Powered by Machine Learning Technology</div>
    <div style='font-size: 0.85rem;'>© 2024 NOUR MAAITA - All Rights Reserved</div>
</div>
""", unsafe_allow_html=True)