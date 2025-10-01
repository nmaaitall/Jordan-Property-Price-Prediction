import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# إعدادات الصفحة
st.set_page_config(
    page_title="Jordan Property Price Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (نفس الكود السابق - محسّن)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    * {
        font-family: 'Cairo', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 50%, #1a1f3a 100%);
    }

    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1500px;
    }

    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 15px rgba(0,0,0,0.4);
    }

    .subtitle {
        text-align: center;
        color: #a8b4c9;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }

    .custom-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(245,247,250,0.95) 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
    }

    .card-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3b82f6;
    }

    .result-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4);
        margin-top: 1.5rem;
    }

    .result-text {
        color: #d1fae5;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-price {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
    }

    .info-badge {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        font-size: 0.9rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e2e8f0;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e3a8a;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
    }

    .trend-up {
        color: #10b981;
        font-weight: 700;
    }

    .trend-down {
        color: #ef4444;
        font-weight: 700;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease;
        margin-top: 1.5rem;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6);
    }

    label {
        font-weight: 700 !important;
        color: #1e293b !important;
        font-size: 1rem !important;
    }

    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1rem !important;
        font-weight: 600 !important;
    }

    .stCheckbox {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 2px solid #e2e8f0;
    }

    .stCheckbox:hover {
        background: #f8fafc;
        border-color: #10b981;
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

    # Dashboard Stats
    if is_arabic:
        st.markdown("### لوحة المؤشرات")
    else:
        st.markdown("### Dashboard")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("العقارات" if is_arabic else "Properties", f"{len(df):,}")
    with col_s2:
        st.metric("المناطق" if is_arabic else "Regions", "20")

    avg_price = df['السعر_دينار'].mean()
    st.metric(
        "متوسط السعر" if is_arabic else "Avg Price",
        f"{avg_price:,.0f}",
        delta="د.أ" if is_arabic else "JOD"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # شرح النموذج
    if is_arabic:
        st.markdown("""
        <div class='info-badge'>
        <strong>كيف يعمل النظام؟</strong><br>
        يستخدم خوارزميات Machine Learning مدربة على 1,500 عقار لتوقع الأسعار بدقة عالية بناءً على 11 عامل مختلف
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-badge'>
        <strong>How It Works?</strong><br>
        Uses ML algorithms trained on 1,500 properties to predict prices accurately based on 11 different factors
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
        st.markdown("<div class='card-header'>🏠 المعلومات الأساسية</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>🏠 Basic Information</div>", unsafe_allow_html=True)

    # نوع العقار
    if is_arabic:
        property_type = st.selectbox("نوع العقار", ["شقة", "فيلا", "أرض"], index=0)
    else:
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Land"], index=0)

    # المنطقة مع بحث
    if is_arabic:
        region_options = regions_ar
        region = st.selectbox("المنطقة", region_options, index=0, help="اختر المنطقة من القائمة أو ابحث")
        region_ar = region
    else:
        region_options = [regions_en[r] for r in regions_ar]
        region = st.selectbox("Region", region_options, index=0, help="Select or search for a region")
        region_ar = [k for k, v in regions_en.items() if v == region][0]

    col_a, col_b = st.columns(2)
    with col_a:
        area = st.number_input("المساحة (م²)" if is_arabic else "Area (sqm)", 50, 1000, 150, 10)
    with col_b:
        rooms = st.number_input("غرف النوم" if is_arabic else "Bedrooms", 1, 10, 3, 1)

    col_c, col_d = st.columns(2)
    with col_c:
        bathrooms = st.number_input("الحمامات" if is_arabic else "Bathrooms", 1, 5, 2, 1)
    with col_d:
        age = st.number_input("عمر البناء" if is_arabic else "Age (years)", 0, 100, 5, 1)

    col_e, col_f = st.columns(2)
    with col_e:
        floor = st.number_input("الطابق" if is_arabic else "Floor", 0, 20, 3, 1)

    services = st.slider("قرب الخدمات" if is_arabic else "Services Proximity", 1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # المميزات
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    if is_arabic:
        st.markdown("<div class='card-header'>✨ المميزات</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>✨ Features</div>", unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        elevator = st.checkbox("مصعد" if is_arabic else "Elevator", True)
        garden = st.checkbox("حديقة" if is_arabic else "Garden", False)
    with col_h:
        parking = st.checkbox("موقف" if is_arabic else "Parking", True)
        heating = st.checkbox("تدفئة" if is_arabic else "Heating", True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    calculate_btn = st.button("احسب السعر" if is_arabic else "Calculate Price", use_container_width=True)

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
                st.markdown(f"<div class='trend-up'>أعلى بـ {abs(diff_percent):.1f}% من متوسط المنطقة</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='trend-down'>أقل بـ {abs(diff_percent):.1f}% من متوسط المنطقة</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-text'>Estimated Price</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} JOD</div>", unsafe_allow_html=True)

            if diff_percent > 0:
                st.markdown(f"<div class='trend-up'>{abs(diff_percent):.1f}% above regional avg</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='trend-down'>{abs(diff_percent):.1f}% below regional avg</div>",
                            unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # رسم بياني محسّن
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
                marker_color=['#10b981', '#64748b'],
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=14, weight=700)
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=300,
            margin=dict(t=40, b=30, l=20, r=20),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            xaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #a8b4c9; font-size: 0.9rem; padding: 1.5rem 0; margin-top: 2rem; border-top: 1px solid rgba(168, 180, 201, 0.2);'>
    <div>Powered by Machine Learning | © 2024 NOUR MAAITA</div>
</div>
""", unsafe_allow_html=True)