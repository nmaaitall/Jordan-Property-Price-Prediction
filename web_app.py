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
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# Custom CSS للتصميم الاحترافي المحسّن
# ═══════════════════════════════════════
st.markdown("""
<style>
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* استيراد خطوط جميلة */
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&family=Poppins:wght@400;600;700;800&display=swap');

    /* الخلفية المتحركة الجميلة */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* إضافة pattern للخلفية */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* Container رئيسي */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }

    /* العنوان الرئيسي المحسّن */
    .main-title {
        text-align: center;
        color: white;
        font-size: 4rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 
            0 0 20px rgba(255, 255, 255, 0.5),
            0 0 40px rgba(255, 255, 255, 0.3),
            3px 3px 8px rgba(0, 0, 0, 0.3);
        animation: titleGlow 3s ease-in-out infinite alternate;
        font-family: 'Tajawal', 'Poppins', sans-serif;
        letter-spacing: 1px;
    }

    @keyframes titleGlow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5), 3px 3px 8px rgba(0, 0, 0, 0.3); }
        to { text-shadow: 0 0 40px rgba(255, 255, 255, 0.8), 0 0 60px rgba(255, 255, 255, 0.4), 3px 3px 8px rgba(0, 0, 0, 0.3); }
    }

    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.4rem;
        margin-bottom: 3rem;
        font-weight: 500;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 1s ease-in-out;
        font-family: 'Tajawal', 'Poppins', sans-serif;
    }

    /* الـ Cards المحسّنة */
    .custom-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.15),
            0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        backdrop-filter: blur(20px);
        margin-bottom: 2rem;
        animation: slideIn 0.8s ease-in-out;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.2);
    }

    /* العناوين داخل الـ Cards */
    .card-header {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-family: 'Tajawal', 'Poppins', sans-serif;
        animation: fadeIn 1s ease-in-out;
    }

    .card-header::before {
        content: '';
        width: 5px;
        height: 35px;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        animation: pulse 2s infinite;
    }

    /* النتيجة المحسّنة */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 30px;
        padding: 3.5rem;
        text-align: center;
        box-shadow: 
            0 20px 60px rgba(102, 126, 234, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.3) inset;
        animation: resultPulse 2.5s infinite;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes resultPulse {
        0%, 100% { transform: scale(1); box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5); }
        50% { transform: scale(1.03); box-shadow: 0 25px 70px rgba(102, 126, 234, 0.7); }
    }

    .result-icon {
        font-size: 6rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
        filter: drop-shadow(0 10px 20px rgba(0, 0, 0, 0.3));
        position: relative;
        z-index: 1;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    .result-text {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: 'Tajawal', 'Poppins', sans-serif;
        position: relative;
        z-index: 1;
    }

    .result-price {
        color: white;
        font-size: 4.5rem;
        font-weight: 900;
        text-shadow: 
            0 0 20px rgba(255, 255, 255, 0.5),
            3px 3px 8px rgba(0, 0, 0, 0.4);
        font-family: 'Tajawal', 'Poppins', sans-serif;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }

    /* الأزرار المحسّنة */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.4rem;
        font-weight: 800;
        padding: 1.3rem 2rem;
        border-radius: 20px;
        border: none;
        box-shadow: 
            0 10px 30px rgba(102, 126, 234, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.3) inset;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-top: 1rem;
        font-family: 'Tajawal', 'Poppins', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }

    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }

    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }

    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 
            0 15px 40px rgba(102, 126, 234, 0.7),
            0 0 0 1px rgba(255, 255, 255, 0.5) inset;
    }

    .stButton>button:active {
        transform: translateY(-2px) scale(1.02);
    }

    /* Inputs محسّنة */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1.5rem;
    }

    /* تحسين شكل الـ input boxes */
    input, select {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
        font-family: 'Tajawal', 'Poppins', sans-serif !important;
    }

    input:focus, select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }

    /* Labels محسّنة */
    label {
        font-weight: 700 !important;
        color: #2d3748 !important;
        font-size: 1.15rem !important;
        font-family: 'Tajawal', 'Poppins', sans-serif !important;
        margin-bottom: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* Sidebar محسّن */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 0 25px 25px 0 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.1) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.5rem !important;
    }

    /* Metrics في الـ Sidebar */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #667eea !important;
        font-family: 'Tajawal', 'Poppins', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-40px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scaleY(1); }
        50% { transform: scaleY(1.2); }
    }

    /* Checkboxes محسّنة */
    .stCheckbox {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .stCheckbox:hover {
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
        border-color: #667eea;
        transform: translateX(5px);
    }

    /* Info boxes محسّنة */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.8rem;
        border-radius: 20px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        font-family: 'Tajawal', 'Poppins', sans-serif;
    }

    .info-box:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }

    /* Slider محسّن */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }

    /* Radio buttons محسّنة */
    .stRadio > div {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1rem;
        border-radius: 15px;
    }

    /* تحسين الـ Plotly charts */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
    }

    /* Divider محسّن */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* تأثير للكروت الفارغة */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    .empty-state-icon {
        font-size: 6rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 10px 30px rgba(102, 126, 234, 0.3));
    }

    /* تحسين scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #667eea);
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
# Sidebar - اختيار اللغة
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌐 Language / اللغة")
    language = st.radio("", ["العربية 🇯🇴", "English 🇬🇧"], label_visibility="collapsed")

    is_arabic = language.startswith("العربية")

    st.markdown("---")

    if is_arabic:
        st.markdown("### 📊 إحصائيات سريعة")
        st.metric("🏘️ عدد العقارات", f"{len(df):,}")
        st.metric("💰 متوسط السعر", f"{df['السعر_دينار'].mean():,.0f} د.أ")
        st.metric("👑 أغلى منطقة", "عبدون")
    else:
        st.markdown("### 📊 Quick Stats")
        st.metric("🏘️ Total Properties", f"{len(df):,}")
        st.metric("💰 Average Price", f"{df['السعر_دينار'].mean():,.0f} JOD")
        st.metric("👑 Most Expensive", "Abdoun")

    st.markdown("---")

    if is_arabic:
        st.markdown("""
        <div class='info-box'>
        <strong>💡 نصيحة احترافية</strong><br><br>
        النموذج مدرّب على 1,500 عقار من 20 منطقة مختلفة في الأردن باستخدام تقنيات الذكاء الاصطناعي المتقدمة
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
        <strong>💡 Professional Tip</strong><br><br>
        Model trained on 1,500 properties from 20 different regions in Jordan using advanced AI techniques
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# العنوان الرئيسي
# ═══════════════════════════════════════
if is_arabic:
    st.markdown("<h1 class='main-title'>🏠 نظام توقع أسعار العقارات في الأردن</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>✨ احصل على تقدير دقيق لسعر العقار باستخدام أحدث تقنيات الذكاء الاصطناعي ✨</p>",
                unsafe_allow_html=True)
else:
    st.markdown("<h1 class='main-title'>🏠 Jordan Property Price Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>✨ Get accurate property price estimates using cutting-edge AI technology ✨</p>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════
# Layout رئيسي
# ═══════════════════════════════════════
col1, col2 = st.columns([1.2, 1])

# ═══════════════════════════════════════
# العمود الأيسر - المدخلات
# ═══════════════════════════════════════
with col1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("<div class='card-header'>📋 معلومات العقار الأساسية</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>📋 Basic Property Information</div>", unsafe_allow_html=True)

    # المنطقة
    if is_arabic:
        region_options = regions_ar
        region = st.selectbox("📍 المنطقة / الموقع", region_options, index=0)
        region_ar = region
    else:
        region_options = [regions_en[r] for r in regions_ar]
        region = st.selectbox("📍 Region / Location", region_options, index=0)
        region_ar = [k for k, v in regions_en.items() if v == region][0]

    # صف المساحة والغرف
    col_a, col_b = st.columns(2)
    with col_a:
        area = st.number_input("📐 " + ("المساحة (متر مربع)" if is_arabic else "Area (sqm)"),
                               min_value=50, max_value=1000, value=150, step=10)
    with col_b:
        rooms = st.number_input("🛏️ " + ("عدد غرف النوم" if is_arabic else "Number of Bedrooms"),
                                min_value=1, max_value=10, value=3, step=1)

    # صف الحمامات والعمر
    col_c, col_d = st.columns(2)
    with col_c:
        bathrooms = st.number_input("🚿 " + ("عدد الحمامات" if is_arabic else "Number of Bathrooms"),
                                    min_value=1, max_value=5, value=2, step=1)
    with col_d:
        age = st.number_input("🏗️ " + ("عمر البناء (سنة)" if is_arabic else "Building Age (years)"),
                              min_value=0, max_value=100, value=5, step=1)

    # صف الطابق والخدمات
    col_e, col_f = st.columns(2)
    with col_e:
        floor = st.number_input("🏢 " + ("رقم الطابق" if is_arabic else "Floor Number"),
                                min_value=0, max_value=20, value=3, step=1)
    with col_f:
        services = st.slider("🏪 " + ("قرب الخدمات (1-10)" if is_arabic else "Proximity to Services (1-10)"),
                             1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # المميزات
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("<div class='card-header'>✨ المميزات والإضافات</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>✨ Features & Amenities</div>", unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        elevator = st.checkbox("🛗 " + ("يوجد مصعد" if is_arabic else "Elevator Available"), value=True)
        garden = st.checkbox("🌳 " + ("يوجد حديقة" if is_arabic else "Garden Available"), value=False)
    with col_h:
        parking = st.checkbox("🚗 " + ("يوجد موقف سيارات" if is_arabic else "Parking Available"), value=True)
        heating = st.checkbox("🔥 " + ("تدفئة مركزية" if is_arabic else "Central Heating"), value=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# العمود الأيمن - النتيجة
# ═══════════════════════════════════════
with col2:
    # زر الحساب
    if is_arabic:
        calculate_btn = st.button("💎 احسب السعر المتوقع", use_container_width=True)
    else:
        calculate_btn = st.button("💎 Calculate Estimated Price", use_container_width=True)

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
        st.markdown("<div class='result-icon'>💰</div>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='result-text'>السعر المتوقع للعقار</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} دينار</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-text'>Estimated Property Price</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} JOD</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # رسم بياني
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='card-header'>📊 مقارنة الأسعار</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-header'>📊 Price Comparison</div>", unsafe_allow_html=True)

        # متوسط أسعار المنطقة
        region_avg = df[df['المنطقة'] == region_ar]['السعر_دينار'].mean()

        fig = go.Figure(data=[
            go.Bar(
                x=[('سعرك المتوقع' if is_arabic else 'Your Price'),
                   ('متوسط المنطقة' if is_arabic else 'Region Average')],
                y=[predicted_price, region_avg],
                marker=dict(
                    color=['#667eea', '#764ba2'],
                    line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
                ),
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=16, color='#2d3748', family='Tajawal, Poppins', weight='bold'),
                hovertemplate='<b>%{x}</b><br>السعر: %{y:,.0f} دينار<extra></extra>'
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4a5568', size=13, family='Tajawal, Poppins'),
            height=350,
            margin=dict(t=30, b=30, l=20, r=20),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)',
                showticklabels=False
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=14, weight='bold')
            ),
            hoverlabel=dict(
                bgcolor='white',
                font_size=14,
                font_family='Tajawal, Poppins'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # معلومات إضافية
        price_diff = predicted_price - region_avg
        price_diff_percent = (price_diff / region_avg) * 100

        if is_arabic:
            if price_diff > 0:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                            color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                            font-weight: 600; font-size: 1.1rem; margin-top: 1rem;
                            box-shadow: 0 5px 15px rgba(72, 187, 120, 0.3);'>
                    📈 سعرك أعلى من متوسط المنطقة بنسبة {abs(price_diff_percent):.1f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); 
                            color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                            font-weight: 600; font-size: 1.1rem; margin-top: 1rem;
                            box-shadow: 0 5px 15px rgba(66, 153, 225, 0.3);'>
                    📉 سعرك أقل من متوسط المنطقة بنسبة {abs(price_diff_percent):.1f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            if price_diff > 0:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                            color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                            font-weight: 600; font-size: 1.1rem; margin-top: 1rem;
                            box-shadow: 0 5px 15px rgba(72, 187, 120, 0.3);'>
                    📈 Your price is {abs(price_diff_percent):.1f}% above the region average
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); 
                            color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                            font-weight: 600; font-size: 1.1rem; margin-top: 1rem;
                            box-shadow: 0 5px 15px rgba(66, 153, 225, 0.3);'>
                    📉 Your price is {abs(price_diff_percent):.1f}% below the region average
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # الحالة الأولية
        st.markdown("<div class='custom-card empty-state'>", unsafe_allow_html=True)
        st.markdown("<div class='empty-state-icon'>🏡</div>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("""
            <div style='font-size: 1.8rem; color: #667eea; font-weight: 700; 
                        font-family: Tajawal, Poppins; margin-bottom: 1rem;'>
                ابدأ بإدخال معلومات العقار
            </div>
            <div style='color: #718096; font-size: 1.1rem; line-height: 1.8;
                        font-family: Tajawal, Poppins;'>
                أدخل تفاصيل العقار من القائمة على اليسار<br>
                ثم اضغط على زر <strong>احسب السعر المتوقع</strong><br>
                للحصول على تقدير فوري ودقيق 🎯
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='font-size: 1.8rem; color: #667eea; font-weight: 700; 
                        font-family: Tajawal, Poppins; margin-bottom: 1rem;'>
                Start by Entering Property Details
            </div>
            <div style='color: #718096; font-size: 1.1rem; line-height: 1.8;
                        font-family: Tajawal, Poppins;'>
                Enter property details from the left panel<br>
                Then click <strong>Calculate Estimated Price</strong><br>
                to get an instant and accurate estimate 🎯
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# قسم معلومات إضافية
# ═══════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div class='custom-card' style='text-align: center; padding: 2rem;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🎯</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;'>
            دقة عالية
        </div>
        <div style='color: #718096; font-size: 0.95rem;'>
            نموذج مدرب على بيانات حقيقية
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class='custom-card' style='text-align: center; padding: 2rem;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>⚡</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;'>
            نتائج فورية
        </div>
        <div style='color: #718096; font-size: 0.95rem;'>
            احصل على التقدير في ثوانٍ
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div class='custom-card' style='text-align: center; padding: 2rem;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;'>
            ذكاء اصطناعي
        </div>
        <div style='color: #718096; font-size: 0.95rem;'>
            تقنية متطورة ومتقدمة
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# Footer محسّن
# ═══════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); 
            border-radius: 20px; backdrop-filter: blur(10px);'>
    <div style='color: white; opacity: 0.9; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;
                font-family: Tajawal, Poppins;'>
        Made with ❤️ using Streamlit & Machine Learning
    </div>
    <div style='color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;'>
        © 2024 Jordan Property Price Predictor | All Rights Reserved
    </div>
</div>
""", unsafe_allow_html=True)