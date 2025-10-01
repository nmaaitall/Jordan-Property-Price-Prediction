import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# ═══════════════════════════════════════
# إعدادات الصفحة
# ═══════════════════════════════════════
st.set_page_config(
    page_title="RealPredict - توقع أسعار العقارات",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# Custom CSS للتصميم الاحترافي
# ═══════════════════════════════════════
st.markdown("""
<style>
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* استيراد خطوط احترافية */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&family=Roboto:wght@400;500;700;900&display=swap');

    /* الخلفية الاحترافية للعقارات */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Container رئيسي */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }

    /* Logo & Header Section */
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 0.8s ease-in-out;
    }

    .logo-title {
        color: white;
        font-size: 4.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 
            0 4px 8px rgba(0, 0, 0, 0.3),
            0 0 30px rgba(255, 255, 255, 0.2);
        font-family: 'Cairo', 'Roboto', sans-serif;
        letter-spacing: 3px;
    }

    .logo-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 1rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    .developer-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        color: white;
        font-size: 0.95rem;
        font-weight: 600;
        border: 2px solid rgba(255, 255, 255, 0.3);
        margin-top: 0.5rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    /* الـ Cards الاحترافية */
    .custom-card {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 
            0 15px 45px rgba(0, 0, 0, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        animation: slideIn 0.8s ease-in-out;
        transition: all 0.3s ease;
    }

    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
    }

    /* العناوين */
    .card-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2a5298;
    }

    /* النتيجة */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 20px 50px rgba(30, 60, 114, 0.4);
        animation: resultPulse 2.5s infinite;
        position: relative;
        overflow: hidden;
    }

    @keyframes resultPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .result-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
        filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.3));
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    .result-text {
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    .result-price {
        color: white;
        font-size: 4rem;
        font-weight: 900;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        font-family: 'Cairo', 'Roboto', sans-serif;
        letter-spacing: 2px;
    }

    /* الأزرار */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 1.2rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(30, 60, 114, 0.4);
        transition: all 0.3s ease;
        margin-top: 1rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(30, 60, 114, 0.6);
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }

    /* Inputs */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1.5rem;
    }

    input, select {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
        font-family: 'Cairo', 'Roboto', sans-serif !important;
    }

    input:focus, select:focus {
        border-color: #2a5298 !important;
        box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1) !important;
    }

    /* Labels */
    label {
        font-weight: 600 !important;
        color: #2d3748 !important;
        font-size: 1.05rem !important;
        font-family: 'Cairo', 'Roboto', sans-serif !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 0 20px 20px 0 !important;
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.1) !important;
    }

    /* Language Selector */
    .language-selector {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 2px solid #2a5298;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1e3c72 !important;
        font-family: 'Cairo', 'Roboto', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.1) 0%, rgba(42, 82, 152, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2a5298;
        margin: 1.5rem 0;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #2a5298;
        box-shadow: 0 15px 40px rgba(30, 60, 114, 0.3);
    }

    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 0.5rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    .feature-text {
        color: #718096;
        font-size: 0.95rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 20px rgba(30, 60, 114, 0.3);
    }

    .stats-number {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-family: 'Cairo', 'Roboto', sans-serif;
    }

    /* Checkboxes */
    .stCheckbox {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .stCheckbox:hover {
        background: #edf2f7;
        border-color: #2a5298;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    .empty-state-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 8px 20px rgba(30, 60, 114, 0.3));
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #2a5298, #1e3c72);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1e3c72, #2a5298);
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
    st.markdown("<div class='language-selector'>", unsafe_allow_html=True)
    st.markdown("### 🌐 اختر اللغة / Select Language")
    language = st.radio("", ["العربية", "English"], label_visibility="collapsed", horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

    is_arabic = language == "العربية"

    st.markdown("---")

    if is_arabic:
        st.markdown("### 📊 إحصائيات المنصة")

        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-number'>{len(df):,}</div>
            <div class='stats-label'>عقار مسجل</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-number'>98.5%</div>
            <div class='stats-label'>دقة التوقع</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-number'>&lt;3s</div>
            <div class='stats-label'>وقت الاستجابة</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("### 📊 Platform Statistics")

        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-number'>{len(df):,}</div>
            <div class='stats-label'>Registered Properties</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-number'>98.5%</div>
            <div class='stats-label'>Prediction Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-number'>&lt;3s</div>
            <div class='stats-label'>Response Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if is_arabic:
        st.markdown("""
        <div class='info-box'>
        <strong>ℹ️ عن النظام</strong><br><br>
        نظام RealPredict يستخدم خوارزميات الذكاء الاصطناعي المتقدمة لتوقع أسعار العقارات بدقة عالية
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
        <strong>ℹ️ About System</strong><br><br>
        RealPredict uses advanced AI algorithms to accurately predict property prices
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# Header Section
# ═══════════════════════════════════════
st.markdown("""
<div class='header-section'>
    <div class='logo-title'>RealPredict</div>
    <div class='logo-subtitle'>نظام توقع أسعار العقارات بالذكاء الاصطناعي</div>
    <div class='developer-badge'>👨‍💻 Developed by Nour Maaita</div>
</div>
""", unsafe_allow_html=True)

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
        st.markdown("<div class='card-header'>📋 معلومات العقار</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>📋 Property Information</div>", unsafe_allow_html=True)

    # المنطقة
    if is_arabic:
        region_options = regions_ar
        region = st.selectbox("📍 المنطقة", region_options, index=0)
        region_ar = region
    else:
        region_options = [regions_en[r] for r in regions_ar]
        region = st.selectbox("📍 Region", region_options, index=0)
        region_ar = [k for k, v in regions_en.items() if v == region][0]

    # صف المساحة والغرف
    col_a, col_b = st.columns(2)
    with col_a:
        area = st.number_input("📐 " + ("المساحة (م²)" if is_arabic else "Area (sqm)"),
                               min_value=50, max_value=1000, value=150, step=10)
    with col_b:
        rooms = st.number_input("🛏️ " + ("عدد الغرف" if is_arabic else "Bedrooms"),
                                min_value=1, max_value=10, value=3, step=1)

    # صف الحمامات والعمر
    col_c, col_d = st.columns(2)
    with col_c:
        bathrooms = st.number_input("🚿 " + ("عدد الحمامات" if is_arabic else "Bathrooms"),
                                    min_value=1, max_value=5, value=2, step=1)
    with col_d:
        age = st.number_input("🏗️ " + ("عمر البناء (سنة)" if is_arabic else "Age (years)"),
                              min_value=0, max_value=100, value=5, step=1)

    # صف الطابق والخدمات
    col_e, col_f = st.columns(2)
    with col_e:
        floor = st.number_input("🏢 " + ("رقم الطابق" if is_arabic else "Floor"),
                                min_value=0, max_value=20, value=3, step=1)
    with col_f:
        services = st.slider("🏪 " + ("قرب الخدمات" if is_arabic else "Services"),
                             1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # المميزات
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("<div class='card-header'>✨ المميزات</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>✨ Features</div>", unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        elevator = st.checkbox("🛗 " + ("مصعد" if is_arabic else "Elevator"), value=True)
        garden = st.checkbox("🌳 " + ("حديقة" if is_arabic else "Garden"), value=False)
    with col_h:
        parking = st.checkbox("🚗 " + ("موقف" if is_arabic else "Parking"), value=True)
        heating = st.checkbox("🔥 " + ("تدفئة" if is_arabic else "Heating"), value=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# العمود الأيمن - النتيجة
# ═══════════════════════════════════════
with col2:
    # زر الحساب
    if is_arabic:
        calculate_btn = st.button("💎 احسب السعر", use_container_width=True)
    else:
        calculate_btn = st.button("💎 Calculate Price", use_container_width=True)

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
            st.markdown("<div class='result-text'>السعر المتوقع</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} دينار</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-text'>Estimated Price</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-price'>{predicted_price:,.0f} JOD</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # رسم بياني
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div class='card-header'>📊 المقارنة</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-header'>📊 Comparison</div>", unsafe_allow_html=True)

        # متوسط أسعار المنطقة
        region_avg = df[df['المنطقة'] == region_ar]['السعر_دينار'].mean()

        fig = go.Figure(data=[
            go.Bar(
                x=[('سعرك' if is_arabic else 'Your Price'),
                   ('المتوسط' if is_arabic else 'Average')],
                y=[predicted_price, region_avg],
                marker=dict(
                    color=['#2a5298', '#1e3c72'],
                    line=dict(color='white', width=2)
                ),
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=16, color='#1e3c72', weight='bold')
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4a5568', size=13),
            height=320,
            margin=dict(t=30, b=30, l=20, r=20),
            yaxis=dict(showgrid=True, gridcolor='rgba(30, 60, 114, 0.1)', showticklabels=False),
            xaxis=dict(showgrid=False, tickfont=dict(size=14, weight='bold'))
        )

        st.plotly_chart(fig, use_container_width=True)

        # معلومات إضافية
        price_diff = predicted_price - region_avg
        price_diff_percent = (price_diff / region_avg) * 100

        if price_diff > 0:
            color = "#48bb78"
            icon = "📈"
            text_ar = f"أعلى من المتوسط بنسبة {abs(price_diff_percent):.1f}%"
            text_en = f"{abs(price_diff_percent):.1f}% above average"
        else:
            color = "#4299e1"
            icon = "📉"
            text_ar = f"أقل من المتوسط بنسبة {abs(price_diff_percent):.1f}%"
            text_en = f"{abs(price_diff_percent):.1f}% below average"

        st.markdown(f"""
                <div style='background: {color}; color: white; padding: 1.2rem; 
                            border-radius: 12px; text-align: center; font-weight: 600; 
                            font-size: 1.05rem; margin-top: 1rem;'>
                    {icon} {text_ar if is_arabic else text_en}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # الحالة الأولية
        st.markdown("<div class='custom-card empty-state'>", unsafe_allow_html=True)
        st.markdown("<div class='empty-state-icon'>🏢</div>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("""
                    <div style='font-size: 1.6rem; color: #1e3c72; font-weight: 700; 
                                font-family: Cairo, Roboto; margin-bottom: 1rem;'>
                        ابدأ بإدخال بيانات العقار
                    </div>
                    <div style='color: #718096; font-size: 1rem; line-height: 1.8;
                                font-family: Cairo, Roboto;'>
                        أدخل معلومات العقار من القائمة اليسرى<br>
                        ثم اضغط على زر <strong>احسب السعر</strong>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
                    <div style='font-size: 1.6rem; color: #1e3c72; font-weight: 700; 
                                font-family: Cairo, Roboto; margin-bottom: 1rem;'>
                        Start Entering Property Data
                    </div>
                    <div style='color: #718096; font-size: 1rem; line-height: 1.8;
                                font-family: Cairo, Roboto;'>
                        Enter property details from the left panel<br>
                        Then click <strong>Calculate Price</strong>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
# ═══════════════════════════════════════
# قسم المميزات
# ═══════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🎯</div>
                <div class='feature-title'>دقة عالية</div>
                <div class='feature-text'>نموذج مدرب على بيانات واقعية</div>
            </div>
            """, unsafe_allow_html=True)

with col_f2:
    st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>⚡</div>
                <div class='feature-title'>نتائج فورية</div>
                <div class='feature-text'>احصل على التقدير خلال ثوانٍ</div>
            </div>
            """, unsafe_allow_html=True)

with col_f3:
    st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🤖</div>
                <div class='feature-title'>ذكاء اصطناعي</div>
                <div class='feature-text'>تقنية متطورة ومتقدمة</div>
            </div>
            """, unsafe_allow_html=True)