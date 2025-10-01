import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# ═══════════════════════════════════════
# إعدادات الصفحة
# ═══════════════════════════════════════
st.set_page_config(
    page_title="Jordan Property Price Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# Custom CSS للتصميم الاحترافي
# ═══════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* الخط الأساسي */
    * {
        font-family: 'Cairo', 'Segoe UI', sans-serif;
    }

    /* الخلفية */
    .stApp {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 50%, #1a1f3a 100%);
        min-height: 100vh;
    }

    /* Container رئيسي */
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1400px;
    }

    /* إزالة الخلفيات البيضاء المزعجة */
    div[data-testid="column"] > div:first-child {
        background: transparent !important;
    }

    div[data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }

    /* العنوان الرئيسي */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 15px rgba(0,0,0,0.4);
    }

    .subtitle {
        text-align: center;
        color: #a8b4c9;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* البطاقات */
    .custom-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(245,247,250,0.95) 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }

    /* عناوين البطاقات */
    .card-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3b82f6;
        display: inline-block;
    }

    /* بطاقة النتيجة */
    .result-card {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
        margin-top: 1.5rem;
    }

    .result-text {
        color: #dbeafe;
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
        letter-spacing: -1px;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
    }

    .result-currency {
        color: #dbeafe;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* الأزرار */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        margin-top: 1.5rem;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }

    /* تحسين Labels */
    label {
        font-weight: 700 !important;
        color: #1e293b !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }

    /* تحسين Selectbox */
    div[data-baseweb="select"] {
        background: white !important;
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    div[data-baseweb="select"]:hover {
        border-color: #3b82f6 !important;
    }

    div[data-baseweb="select"] > div {
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* تحسين Number Input */
    .stNumberInput > div > div > input {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* أزرار Number Input */
    .stNumberInput button {
        background: #f1f5f9 !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        width: 35px !important;
        height: 35px !important;
    }

    .stNumberInput button:hover {
        background: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }

    /* تحسين Slider */
    .stSlider {
        padding: 1rem 0.5rem !important;
    }

    .stSlider > div > div > div > div {
        background: #3b82f6 !important;
    }

    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid #3b82f6 !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }

    /* تحسين Checkboxes */
    .stCheckbox {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 2px solid #e2e8f0;
        transition: all 0.2s ease;
    }

    .stCheckbox:hover {
        background: #f8fafc;
        border-color: #3b82f6;
        transform: translateX(-2px);
    }

    .stCheckbox label {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 0.95rem !important;
    }

    .stCheckbox input[type="checkbox"]:checked ~ label {
        color: #3b82f6 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #f8fafc, #e2e8f0);
        border-right: 1px solid #cbd5e1;
    }

    [data-testid="stSidebar"] .element-container {
        padding: 0.3rem 0;
    }

    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #1e293b;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .info-box strong {
        color: #1e3a8a;
        display: block;
        margin-bottom: 0.3rem;
        font-size: 1rem;
    }

    /* Stats Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #1e3a8a !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Initial State */
    .initial-state {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.95));
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        border: 2px dashed #cbd5e1;
        margin-top: 1.5rem;
    }

    .initial-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }

    .initial-state-title {
        font-size: 1.4rem;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .initial-state-desc {
        color: #64748b;
        font-size: 1rem;
    }

    /* Radio Buttons */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }

    .stRadio label {
        font-weight: 600 !important;
        color: #1e293b !important;
    }

    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }

    /* تحسين العرض */
    .row-widget {
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# تحميل النموذج
# ═══════════════════════════════════════
@st.cache_resource
def load_model():
    """تحميل النموذج والبيانات"""
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

# ═══════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("### اللغة / Language")
    language = st.radio("", ["العربية", "English"], label_visibility="collapsed")

    is_arabic = language == "العربية"

    st.markdown("<hr>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("### إحصائيات النظام")
        st.metric("عدد العقارات", f"{len(df):,}")
        st.metric("متوسط السعر", f"{df['السعر_دينار'].mean():,.0f} د.أ")
        st.metric("أعلى منطقة سعراً", "عبدون")
    else:
        st.markdown("### System Statistics")
        st.metric("Total Properties", f"{len(df):,}")
        st.metric("Average Price", f"{df['السعر_دينار'].mean():,.0f} JOD")
        st.metric("Highest Priced Area", "Abdoun")

    st.markdown("<hr>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("""
        <div class='info-box'>
        <strong>معلومات النموذج</strong>
        نموذج التعلم الآلي مدرب على 1,500 عقار من 20 منطقة مختلفة في الأردن بدقة عالية
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
        <strong>Model Information</strong>
        Machine learning model trained on 1,500 properties from 20 different regions in Jordan with high accuracy
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# العنوان الرئيسي
# ═══════════════════════════════════════
if is_arabic:
    st.markdown("<h1 class='main-title'>نظام التنبؤ بأسعار العقارات في الأردن</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>احصل على تقييم دقيق لسعر العقار باستخدام تقنيات الذكاء الاصطناعي المتقدمة</p>",
                unsafe_allow_html=True)
else:
    st.markdown("<h1 class='main-title'>Jordan Property Price Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Get accurate property valuation using advanced AI technology</p>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════
# Layout رئيسي
# ═══════════════════════════════════════
col1, col2 = st.columns([1.4, 1], gap="large")

# ═══════════════════════════════════════
# العمود الأيسر - المدخلات
# ═══════════════════════════════════════
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

    st.markdown("<br>", unsafe_allow_html=True)

    # صف المساحة والغرف
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        area = st.number_input("المساحة (متر مربع)" if is_arabic else "Area (sqm)",
                               min_value=50, max_value=1000, value=150, step=10)
    with col_b:
        rooms = st.number_input("عدد غرف النوم" if is_arabic else "Number of Bedrooms",
                                min_value=1, max_value=10, value=3, step=1)

    # صف الحمامات والعمر
    col_c, col_d = st.columns(2, gap="medium")
    with col_c:
        bathrooms = st.number_input("عدد الحمامات" if is_arabic else "Number of Bathrooms",
                                    min_value=1, max_value=5, value=2, step=1)
    with col_d:
        age = st.number_input("عمر البناء (سنة)" if is_arabic else "Building Age (years)",
                              min_value=0, max_value=100, value=5, step=1)

    # صف الطابق
    col_e, col_f = st.columns(2, gap="medium")
    with col_e:
        floor = st.number_input("رقم الطابق" if is_arabic else "Floor Number",
                                min_value=0, max_value=20, value=3, step=1)
    with col_f:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Slider للخدمات
    services = st.slider("قرب الخدمات (1-10)" if is_arabic else "Proximity to Services (1-10)",
                         1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # المميزات
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("<div class='card-header'>المميزات والخدمات</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>Features & Amenities</div>", unsafe_allow_html=True)

    col_g, col_h = st.columns(2, gap="medium")
    with col_g:
        elevator = st.checkbox("يوجد مصعد" if is_arabic else "Elevator Available", value=True)
        garden = st.checkbox("يوجد حديقة" if is_arabic else "Garden Available", value=False)
    with col_h:
        parking = st.checkbox("يوجد موقف سيارات" if is_arabic else "Parking Available", value=True)
        heating = st.checkbox("تدفئة مركزية" if is_arabic else "Central Heating", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# العمود الأيمن - النتيجة
# ═══════════════════════════════════════
with col2:
    # زر الحساب
    if is_arabic:
        calculate_btn = st.button("احسب السعر المتوقع", use_container_width=True)
    else:
        calculate_btn = st.button("Calculate Estimated Price", use_container_width=True)

    if calculate_btn:
        # التوقع
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

        # عرض النتيجة
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='result-text'>السعر المتوقع</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='result-currency'>دينار أردني</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-text'>Estimated Price</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='result-currency'>Jordanian Dinar</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # رسم بياني
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='card-header'>مقارنة الأسعار</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-header'>Price Comparison</div>", unsafe_allow_html=True)

        # متوسط أسعار المنطقة
        region_avg = df[df['المنطقة'] == region_ar]['السعر_دينار'].mean()

        fig = go.Figure(data=[
            go.Bar(
                x=[('السعر المتوقع' if is_arabic else 'Estimated Price'),
                   ('متوسط المنطقة' if is_arabic else 'Region Average')],
                y=[predicted_price, region_avg],
                marker_color=['#3b82f6', '#60a5fa'],
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=14, color='#1e293b', family='Cairo', weight=700)
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            font=dict(color='#1e293b', size=12, family='Cairo'),
            height=300,
            margin=dict(t=40, b=30, l=20, r=20),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', showticklabels=True,
                       tickfont=dict(size=11, color='#64748b')),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color='#1e293b', weight=600))
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # الحالة الأولية
        st.markdown("<div class='initial-state'>", unsafe_allow_html=True)
        st.markdown("<div class='initial-state-icon'>🏛️</div>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='initial-state-title'>ابدأ التقييم</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='initial-state-desc'>أدخل معلومات العقار واضغط على زر الحساب للحصول على التقييم</div>",
                unsafe_allow_html=True)
        else:
            st.markdown("<div class='initial-state-title'>Start Valuation</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='initial-state-desc'>Enter property details and click calculate to get your valuation</div>",
                unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# Footer
# ═══════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #a8b4c9; font-size: 0.9rem; padding: 1.5rem 0; border-top: 1px solid rgba(168, 180, 201, 0.2); margin-top: 2rem;'>
    <div style='margin-bottom: 0.5rem;'>Powered by Machine Learning Technology</div>
    <div style='font-size: 0.85rem; opacity: 0.8;'>© 2024 NOUR MAAITA - All Rights Reserved</div>
</div>
""", unsafe_allow_html=True)