import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(
    page_title="RealPredict - Smart Real Estate Valuation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translations
translations = {
    'en': {
        'title': 'RealPredict',
        'subtitle': 'Professional Real Estate Valuation Platform',
        'powered': 'Powered by Advanced Machine Learning & Market Analytics',
        'location': 'Property Location & Details',
        'region': 'Select Region',
        'area': 'Area (Square Meters)',
        'bedrooms': 'Number of Bedrooms',
        'bathrooms': 'Number of Bathrooms',
        'age': 'Property Age (Years)',
        'features': 'Property Features & Amenities',
        'elevator': 'Elevator Available',
        'parking': 'Parking Space',
        'garden': 'Private Garden',
        'heating': 'Central Heating',
        'services': 'Proximity to Services & Amenities (1-10)',
        'calculate': 'Calculate Property Value',
        'estimated': 'Estimated Market Value',
        'currency': 'Jordanian Dinar (JOD)',
        'above': 'Above Regional Average',
        'below': 'Below Regional Average',
        'price_sqm': 'Price/SQM',
        'region_avg': 'Region Avg',
        'comparison': 'Market Comparison Analysis',
        'your_property': 'Your Property',
        'regional_avg': 'Regional Average',
        'ready': 'Ready to Estimate',
        'enter_details': 'Enter property details and click',
        'language': 'Language',
        'theme': 'Theme',
        'light': 'Light',
        'dark': 'Dark',
        'floor': 'Floor Number'
    },
    'ar': {
        'title': 'RealPredict',
        'subtitle': 'منصة احترافية لتقييم العقارات',
        'powered': 'مدعوم بالذكاء الاصطناعي وتحليلات السوق المتقدمة',
        'location': 'الموقع وتفاصيل العقار',
        'region': 'اختر المنطقة',
        'area': 'المساحة (متر مربع)',
        'bedrooms': 'عدد غرف النوم',
        'bathrooms': 'عدد الحمامات',
        'age': 'عمر العقار (سنوات)',
        'features': 'مميزات ووسائل الراحة',
        'elevator': 'يوجد مصعد',
        'parking': 'موقف سيارات',
        'garden': 'حديقة خاصة',
        'heating': 'تدفئة مركزية',
        'services': 'القرب من الخدمات (1-10)',
        'calculate': 'احسب قيمة العقار',
        'estimated': 'القيمة السوقية المقدرة',
        'currency': 'دينار أردني',
        'above': 'أعلى من متوسط المنطقة',
        'below': 'أقل من متوسط المنطقة',
        'price_sqm': 'السعر/متر',
        'region_avg': 'متوسط المنطقة',
        'comparison': 'تحليل مقارنة السوق',
        'your_property': 'عقارك',
        'regional_avg': 'المتوسط الإقليمي',
        'ready': 'جاهز للتقييم',
        'enter_details': 'أدخل تفاصيل العقار واضغط',
        'language': 'اللغة',
        'theme': 'المظهر',
        'light': 'فاتح',
        'dark': 'داكن',
        'floor': 'رقم الطابق'
    }
}

# Initialize session state
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Sidebar settings
with st.sidebar:
    st.markdown("### Settings / الإعدادات")

    lang_choice = st.radio(
        "Language / اللغة",
        options=['en', 'ar'],
        format_func=lambda x: 'English' if x == 'en' else 'العربية',
        index=0 if st.session_state.lang == 'en' else 1,
        horizontal=True
    )

    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    theme_choice = st.radio(
        "Theme / المظهر",
        options=['light', 'dark'],
        format_func=lambda x: 'Light / فاتح' if x == 'light' else 'Dark / داكن',
        index=0 if st.session_state.theme == 'light' else 1,
        horizontal=True
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

lang = st.session_state.lang
theme = st.session_state.theme
t = translations[lang]

# Dynamic CSS
bg_gradient = 'linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%)' if theme == 'light' else 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)'
card_bg = '#ffffff' if theme == 'light' else '#1e293b'
text_primary = '#1e3c72' if theme == 'light' else '#f1f5f9'
text_secondary = '#334155' if theme == 'light' else '#cbd5e1'
border_color = '#e8ecf1' if theme == 'light' else '#334155'
input_bg = '#f8fafc' if theme == 'light' else '#0f172a'
input_border = '#e2e8f0' if theme == 'light' else '#475569'
hover_bg = '#f1f5f9' if theme == 'light' else '#334155'
checkbox_bg = '#f8fafc' if theme == 'light' else '#1e293b'
checkbox_text = '#334155' if theme == 'light' else '#e2e8f0'

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: {bg_gradient};
    }}

    section[data-testid="stSidebar"] {{
        background: {card_bg};
        border-right: 2px solid {border_color};
    }}

    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {text_primary} !important;
        font-size: 1.2rem !important;
    }}

    section[data-testid="stSidebar"] label {{
        color: {text_secondary} !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }}

    section[data-testid="stSidebar"] .stRadio > label {{
        color: {text_primary} !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: {input_bg} !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        border: 2px solid {input_border} !important;
        margin: 0.25rem 0 !important;
        transition: all 0.3s ease !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        border-color: #2a5298 !important;
        background: {hover_bg} !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
        background: #2a5298 !important;
        color: white !important;
        border-color: #2a5298 !important;
    }}

    .main .block-container {{
        padding: 2rem 3rem;
        max-width: 1400px;
    }}

    .header-container {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px rgba(30, 60, 114, 0.25);
        position: relative;
        overflow: hidden;
    }}

    .header-container::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }}

    .logo-container {{
        position: relative;
        z-index: 1;
        text-align: {'right' if lang == 'ar' else 'left'};
    }}

    .logo-text {{
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
        letter-spacing: -1px;
    }}

    .logo-icon {{
        display: inline-block;
        margin-{'left' if lang == 'ar' else 'right'}: 0.5rem;
        font-size: 3rem;
    }}

    .tagline {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.15rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}

    .subtitle {{
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.95rem;
        margin-top: 1rem;
        font-weight: 400;
    }}

    .input-section {{
        background: {card_bg};
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, {'0.08' if theme == 'light' else '0.3'});
        border: 1px solid {border_color};
        margin-bottom: 1.5rem;
    }}

    .section-header {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #2a5298;
        display: flex;
        align-items: center;
        direction: {'rtl' if lang == 'ar' else 'ltr'};
    }}

    .section-icon {{
        margin-{'left' if lang == 'ar' else 'right'}: 0.75rem;
        font-size: 1.6rem;
    }}

    .result-card {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(30, 60, 114, 0.35);
        position: relative;
        overflow: hidden;
        min-height: 380px;
    }}

    .result-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(1.1); opacity: 0.3; }}
    }}

    .result-content {{
        position: relative;
        z-index: 1;
    }}

    .result-badge {{
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
    }}

    .result-price {{
        color: #ffffff;
        font-size: 4.5rem;
        font-weight: 900;
        margin: 1.5rem 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        line-height: 1;
        letter-spacing: -2px;
    }}

    .result-currency {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }}

    .trend-container {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1.25rem 2rem;
        margin: 2rem auto 0 auto;
        max-width: 85%;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }}

    .trend-content {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }}

    .trend-icon {{
        font-size: 2rem;
        font-weight: 800;
    }}

    .trend-text {{
        color: #1e293b;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }}

    label {{
        font-weight: 600 !important;
        color: {text_secondary} !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }}

    div[data-baseweb="select"] {{
        background: {input_bg} !important;
        border-radius: 12px !important;
        border: 2px solid {input_border} !important;
        transition: all 0.3s ease !important;
    }}

    div[data-baseweb="select"]:hover {{
        border-color: #2a5298 !important;
        background: {hover_bg} !important;
    }}

    div[data-baseweb="select"] > div {{
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {text_primary} !important;
        background: transparent !important;
    }}

    div[data-baseweb="select"] svg {{
        fill: {text_primary} !important;
    }}

    div[data-baseweb="popover"] {{
        background: {card_bg} !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, {'0.15' if theme == 'light' else '0.5'}) !important;
        border: 2px solid {input_border} !important;
        border-radius: 12px !important;
    }}

    ul[role="listbox"] {{
        background: {card_bg} !important;
        border: none !important;
    }}

    ul[role="listbox"] li {{
        color: {text_primary} !important;
        background: {card_bg} !important;
        padding: 1rem 1.25rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}

    ul[role="listbox"] li:hover {{
        background: {hover_bg} !important;
        color: #2a5298 !important;
    }}

    ul[role="listbox"] li[aria-selected="true"] {{
        background: #2a5298 !important;
        color: #ffffff !important;
    }}

    .stNumberInput > div > div > input {{
        background: {input_bg} !important;
        border: 2px solid {input_border} !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {text_primary} !important;
        transition: all 0.3s ease !important;
    }}

    .stNumberInput > div > div > input:focus {{
        border-color: #2a5298 !important;
        box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1) !important;
        background: {card_bg} !important;
    }}

    .stSlider > div > div > div > div {{
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%) !important;
        height: 6px !important;
    }}

    .stSlider > div > div > div > div > div {{
        background: white !important;
        border: 4px solid #2a5298 !important;
        width: 26px !important;
        height: 26px !important;
        box-shadow: 0 3px 12px rgba(42, 82, 152, 0.4) !important;
    }}

    .stCheckbox {{
        background: {checkbox_bg};
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border: 2px solid {input_border};
        transition: all 0.3s ease;
    }}

    .stCheckbox:hover {{
        border-color: #2a5298;
        background: {hover_bg};
    }}

    .stCheckbox label {{
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: {checkbox_text} !important;
    }}

    .stCheckbox label span {{
        color: {checkbox_text} !important;
    }}

    .stCheckbox label p {{
        color: {checkbox_text} !important;
    }}

    .stButton>button {{
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
    }}

    .stButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(42, 82, 152, 0.45);
    }}

    .chart-section {{
        background: {card_bg};
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, {'0.08' if theme == 'light' else '0.3'});
        border: 1px solid {border_color};
    }}

    .chart-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 1.5rem;
    }}

    .info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }}

    .info-card-mini {{
        background: {input_bg};
        border-radius: 12px;
        padding: 1.25rem;
        border: 2px solid {input_border};
        text-align: center;
    }}

    .info-card-mini .label {{
        font-size: 0.85rem;
        color: {text_secondary};
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .info-card-mini .value {{
        font-size: 1.5rem;
        color: {text_primary};
        font-weight: 800;
    }}
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
st.markdown(f"""
<div class='header-container'>
    <div class='logo-container'>
        <div class='logo-text'>
            <span class='logo-icon'>🏢</span>{t['title']}
        </div>
        <div class='tagline'>{t['subtitle']}</div>
        <div class='subtitle'>{t['powered']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Layout
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown(
        f"<div class='input-section'><div class='section-header'><span class='section-icon'>📍</span>{t['location']}</div>",
        unsafe_allow_html=True)

    # Fixed selectbox - using placeholder to show selected value
    if lang == 'ar':
        region_options = regions_ar
    else:
        region_options = [regions_en[r] for r in regions_ar]

    region = st.selectbox(
        t['region'],
        options=region_options,
        index=0
    )

    # Convert to Arabic region name
    if lang == 'ar':
        region_ar = region
    else:
        region_ar = regions_ar[region_options.index(region)]

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(t['area'], 50, 1000, 150, 10)
        rooms = st.number_input(t['bedrooms'], 1, 10, 3, 1)
    with col2:
        bathrooms = st.number_input(t['bathrooms'], 1, 5, 2, 1)
        age = st.number_input(t['age'], 0, 100, 5, 1)

    floor = st.number_input(t['floor'], 0, 20, 3, 1)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='input-section'><div class='section-header'><span class='section-icon'>✨</span>{t['features']}</div>",
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        elevator = st.checkbox(t['elevator'], value=True)
        parking = st.checkbox(t['parking'], value=True)
    with col2:
        garden = st.checkbox(t['garden'], value=False)
        heating = st.checkbox(t['heating'], value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    services = st.slider(t['services'], 1, 10, 7)

    st.markdown("</div>", unsafe_allow_html=True)

    predict_button = st.button(t['calculate'])

with col_right:
    if predict_button:
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

        trend_icon = '↑' if diff_percent > 0 else '↓'
        trend_color = '#10b981' if diff_percent > 0 else '#ef4444'
        trend_text = t['above'] if diff_percent > 0 else t['below']

        st.markdown(f"""
            <div class='result-card'>
                <div class='result-content'>
                    <div class='result-badge'>{t['estimated']}</div>
                    <div class='result-price'>{predicted_price:,.0f}</div>
                    <div class='result-currency'>{t['currency']}</div>
                    <div class='trend-container'>
                        <div class='trend-content'>
                            <span class='trend-icon' style='color: {trend_color};'>{trend_icon}</span>
                            <p class='trend-text'>{abs(diff_percent):.1f}% {trend_text}</p>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-grid'>
            <div class='info-card-mini'>
                <div class='label'>{t['price_sqm']}</div>
                <div class='value'>{predicted_price / area:,.0f}</div>
            </div>
            <div class='info-card-mini'>
                <div class='label'>{t['region_avg']}</div>
                <div class='value'>{region_avg:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='chart-section'><div class='chart-title'>{t['comparison']}</div>",
                    unsafe_allow_html=True)

        bar_color = '#2a5298' if theme == 'light' else '#60a5fa'
        avg_color = '#94a3b8' if theme == 'light' else '#475569'
        grid_color = '#e2e8f0' if theme == 'light' else '#334155'

        fig = go.Figure(data=[
            go.Bar(
                x=[t['your_property'], t['regional_avg']],
                y=[predicted_price, region_avg],
                marker=dict(
                    color=[bar_color, avg_color],
                    line=dict(color=bar_color, width=2)
                ),
                text=[f'{predicted_price:,.0f}', f'{region_avg:,.0f}'],
                textposition='outside',
                textfont=dict(size=14, weight=700, color=text_primary)
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=280,
            margin=dict(t=40, b=30, l=20, r=20),
            yaxis=dict(
                showgrid=True,
                gridcolor=grid_color,
                showticklabels=True,
                tickfont=dict(size=11, color=text_secondary)
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=12, weight=600, color=text_primary)
            ),
            font=dict(family='Inter, sans-serif')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='result-card'>
                <div class='result-content'>
                    <div class='result-badge'>{t['ready']}</div>
                    <div style='font-size: 4rem; margin: 2rem 0;'>🏢</div>
                    <div class='result-currency' style='margin-bottom: 1rem;'>{t['enter_details']}</div>
                    <div class='result-currency'>"{t['calculate']}"</div>
                </div>
            </div>
        """, unsafe_allow_html=True)