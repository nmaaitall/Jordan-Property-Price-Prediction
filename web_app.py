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
# Custom CSS للتصميم الاحترافي
# ═══════════════════════════════════════
st.markdown("""
<style>
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* الخلفية والألوان */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Container رئيسي */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* العنوان الرئيسي */
    .main-title {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeInDown 0.8s ease-in-out;
    }

    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin-bottom: 3rem;
        animation: fadeInUp 0.8s ease-in-out;
    }

    /* الـ Cards */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in-out;
    }

    /* العناوين داخل الـ Cards */
    .card-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* النتيجة */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        animation: pulse 2s infinite;
    }

    .result-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
    }

    .result-text {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        opacity: 0.9;
    }

    .result-price {
        color: white;
        font-size: 4rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* الأزرار */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }

    /* Inputs */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1rem;
    }

    /* Labels */
    label {
        font-weight: 600 !important;
        color: #4a5568 !important;
        font-size: 1.1rem !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    /* Checkboxes */
    .stCheckbox {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
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
        st.metric("عدد العقارات", f"{len(df):,}")
        st.metric("متوسط السعر", f"{df['السعر_دينار'].mean():,.0f} د.أ")
        st.metric("أغلى منطقة", "عبدون")
    else:
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Properties", f"{len(df):,}")
        st.metric("Average Price", f"{df['السعر_دينار'].mean():,.0f} JOD")
        st.metric("Most Expensive", "Abdoun")

    st.markdown("---")

    if is_arabic:
        st.markdown("""
        <div class='info-box'>
        <strong>💡 نصيحة:</strong><br>
        النموذج مدرّب على 1,500 عقار من 20 منطقة مختلفة في الأردن
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
        <strong>💡 Tip:</strong><br>
        Model trained on 1,500 properties from 20 different regions in Jordan
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# العنوان الرئيسي
# ═══════════════════════════════════════
if is_arabic:
    st.markdown("<h1 class='main-title'>🏠 نظام توقع أسعار العقارات في الأردن</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>احصل على تقدير دقيق لسعر العقار باستخدام الذكاء الاصطناعي</p>",
                unsafe_allow_html=True)
else:
    st.markdown("<h1 class='main-title'>🏠 Jordan Property Price Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Get accurate property price estimates using AI technology</p>",
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
        area = st.number_input("📏 " + ("المساحة (م²)" if is_arabic else "Area (sqm)"),
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
        age = st.number_input("📅 " + ("عمر البناء (سنة)" if is_arabic else "Building Age (years)"),
                              min_value=0, max_value=100, value=5, step=1)

    # صف الطابق والخدمات
    col_e, col_f = st.columns(2)
    with col_e:
        floor = st.number_input("🏢 " + ("رقم الطابق" if is_arabic else "Floor Number"),
                                min_value=0, max_value=20, value=3, step=1)
    with col_f:
        services = st.slider("🏪 " + ("قرب الخدمات" if is_arabic else "Proximity to Services"),
                             1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    # المميزات
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

    if is_arabic:
        st.markdown("<div class='card-header'>✨ المميزات الإضافية</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-header'>✨ Additional Features</div>", unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        elevator = st.checkbox("🛗 " + ("مصعد" if is_arabic else "Elevator"), value=True)
        garden = st.checkbox("🌳 " + ("حديقة" if is_arabic else "Garden"), value=False)
    with col_h:
        parking = st.checkbox("🚗 " + ("موقف" if is_arabic else "Parking"), value=True)
        heating = st.checkbox("🔥 " + ("تدفئة مركزية" if is_arabic else "Central Heating"), value=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# العمود الأيمن - النتيجة
# ═══════════════════════════════════════
with col2:
    # زر الحساب
    if is_arabic:
        calculate_btn = st.button("💰 احسب السعر الآن", use_container_width=True)
    else:
        calculate_btn = st.button("💰 Calculate Price Now", use_container_width=True)

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
        st.markdown("<div class='result-icon'>💵</div>", unsafe_allow_html=True)

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
            st.markdown("<div class='card-header'>📊 مقارنة السعر</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-header'>📊 Price Comparison</div>", unsafe_allow_html=True)

        # متوسط أسعار المنطقة
        region_avg = df[df['المنطقة'] == region_ar]['السعر_دينار'].mean()

        fig = go.Figure(data=[
            go.Bar(
                x=[('سعرك المتوقع' if is_arabic else 'Your Price'),
                   ('متوسط المنطقة' if is_arabic else 'Region Average')],
                y=[predicted_price, region_avg],
                marker_color=['#667eea', '#764ba2'],
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=14, color='white', family='Arial Black')
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4a5568', size=12),
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(showgrid=False, showticklabels=False),
            xaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # الحالة الأولية
        st.markdown("<div class='custom-card' style='text-align: center; padding: 4rem 2rem;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 5rem; margin-bottom: 1rem;'>🏠</div>", unsafe_allow_html=True)

        if is_arabic:
            st.markdown("<div style='font-size: 1.5rem; color: #667eea; font-weight: 600;'>أدخل معلومات العقار</div>",
                        unsafe_allow_html=True)
            st.markdown("<div style='color: #718096; margin-top: 1rem;'>واضغط على زر احسب السعر</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-size: 1.5rem; color: #667eea; font-weight: 600;'>Enter Property Information</div>",
                unsafe_allow_html=True)
            st.markdown("<div style='color: #718096; margin-top: 1rem;'>and click Calculate Price button</div>",
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# Footer
# ═══════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; opacity: 0.8; font-size: 0.9rem;'>
    Made with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)