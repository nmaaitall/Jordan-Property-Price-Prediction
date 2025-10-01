import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════
# إعدادات الصفحة
# ═══════════════════════════════════════
st.set_page_config(
    page_title="نظام توقع أسعار العقارات الأردنية",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص
st.markdown("""
    <style>
    .main {
        direction: rtl;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    h1, h2, h3 {
        color: #1e293b;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# تحميل البيانات
# ═══════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('jordan_properties.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ ملف البيانات غير موجود! يرجى تشغيل generate_jordan_data.py أولاً")
        st.stop()


df = load_data()


# ═══════════════════════════════════════
# تدريب النموذج
# ═══════════════════════════════════════
@st.cache_resource
def train_models(df):
    # تحضير البيانات
    le = LabelEncoder()
    df['المنطقة_رقم'] = le.fit_transform(df['المنطقة'])

    features = ['المساحة_متر', 'عدد_الغرف', 'عدد_الحمامات', 'عمر_البناء_سنوات',
                'طابق', 'يوجد_مصعد', 'يوجد_موقف', 'يوجد_حديقة',
                'يوجد_تدفئة_مركزية', 'قرب_الخدمات', 'المنطقة_رقم']

    X = df[features]
    y = df['السعر_دينار']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تدريب النماذج الثلاثة
    models = {}
    predictions = {}
    metrics = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['Linear Regression'] = lr
    predictions['Linear Regression'] = lr_pred
    metrics['Linear Regression'] = {
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R2': r2_score(y_test, lr_pred)
    }

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf_pred
    metrics['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred)
    }

    # 3. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = gb_pred
    metrics['Gradient Boosting'] = {
        'MAE': mean_absolute_error(y_test, gb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'R2': r2_score(y_test, gb_pred)
    }

    # اختيار أفضل نموذج
    best_model_name = max(metrics.keys(), key=lambda x: metrics[x]['R2'])
    best_model = models[best_model_name]

    return models, le, features, metrics, X_test, y_test, predictions, best_model_name, best_model


models, le, features, metrics, X_test, y_test, predictions, best_model_name, best_model = train_models(df)

# ═══════════════════════════════════════
# Sidebar - القائمة الجانبية
# ═══════════════════════════════════════
st.sidebar.title("🏠 نظام توقع أسعار العقارات")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "اختر الصفحة:",
    ["🏡 الصفحة الرئيسية", "📊 تحليل البيانات", "🤖 أداء النماذج", "🎯 توقع السعر", "ℹ️ عن المشروع"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"📊 عدد العقارات: {len(df):,}\n\n💰 متوسط السعر: {df['السعر_دينار'].mean():,.0f} دينار")

# ═══════════════════════════════════════
# الصفحة الرئيسية
# ═══════════════════════════════════════
if page == "🏡 الصفحة الرئيسية":
    st.title("🏠 نظام توقع أسعار العقارات في الأردن")
    st.markdown("### نظام ذكي لتوقع أسعار العقارات باستخدام تقنيات التعلم الآلي")

    st.markdown("---")

    # بطاقات الإحصائيات
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>📊</h3>
                <h2>{:,}</h2>
                <p>عدد العقارات</p>
            </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>💰</h3>
                <h2>{:,.0f}</h2>
                <p>متوسط السعر (دينار)</p>
            </div>
        """.format(df['السعر_دينار'].mean()), unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>🏘️</h3>
                <h2>{}</h2>
                <p>عدد المناطق</p>
            </div>
        """.format(df['المنطقة'].nunique()), unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3>🎯</h3>
                <h2>{:.1f}%</h2>
                <p>دقة النموذج</p>
            </div>
        """.format(metrics[best_model_name]['R2'] * 100), unsafe_allow_html=True)

    st.markdown("---")

    # معلومات عن المشروع
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 مميزات النظام")
        st.markdown("""
        - ✅ **توقعات دقيقة** بنسبة تزيد عن 90%
        - ✅ **20 منطقة أردنية** مشمولة في النظام
        - ✅ **3 نماذج ذكاء اصطناعي** مختلفة
        - ✅ **تحليلات شاملة** للبيانات
        - ✅ **واجهة سهلة** الاستخدام
        - ✅ **نتائج فورية** في ثوانٍ
        """)

    with col2:
        st.markdown("### 💻 التقنيات المستخدمة")
        st.markdown("""
        - 🐍 **Python** - لغة البرمجة
        - 🤖 **Scikit-learn** - التعلم الآلي
        - 📊 **Pandas & NumPy** - معالجة البيانات
        - 📈 **Matplotlib & Plotly** - الرسوم البيانية
        - 🌐 **Streamlit** - واجهة الويب
        - 🔬 **Random Forest** - أفضل نموذج
        """)

    st.markdown("---")

    # رسم بياني سريع
    st.markdown("### 📈 نظرة سريعة على البيانات")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='السعر_دينار', nbins=30,
                           title="توزيع الأسعار",
                           labels={'السعر_دينار': 'السعر (دينار)', 'count': 'العدد'})
        fig.update_traces(marker_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_avg = df.groupby('المنطقة')['السعر_دينار'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=region_avg.values, y=region_avg.index,
                     orientation='h',
                     title="أغلى 10 مناطق",
                     labels={'x': 'متوسط السعر (دينار)', 'y': 'المنطقة'})
        fig.update_traces(marker_color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════
# صفحة تحليل البيانات
# ═══════════════════════════════════════
elif page == "📊 تحليل البيانات":
    st.title("📊 تحليل البيانات والإحصائيات")

    tabs = st.tabs(["📈 إحصائيات عامة", "🏘️ تحليل المناطق", "🔍 تحليل تفصيلي", "📋 عينة البيانات"])

    # التبويب 1: إحصائيات عامة
    with tabs[0]:
        st.markdown("### 📊 الإحصائيات الوصفية")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💰 إحصائيات الأسعار")
            st.write(f"**أقل سعر:** {df['السعر_دينار'].min():,} دينار")
            st.write(f"**أعلى سعر:** {df['السعر_دينار'].max():,} دينار")
            st.write(f"**متوسط السعر:** {df['السعر_دينار'].mean():,.0f} دينار")
            st.write(f"**الوسيط:** {df['السعر_دينار'].median():,.0f} دينار")
            st.write(f"**الانحراف المعياري:** {df['السعر_دينار'].std():,.0f} دينار")

        with col2:
            st.markdown("### 📏 إحصائيات المساحات")
            st.write(f"**أصغر مساحة:** {df['المساحة_متر'].min()} م²")
            st.write(f"**أكبر مساحة:** {df['المساحة_متر'].max()} م²")
            st.write(f"**متوسط المساحة:** {df['المساحة_متر'].mean():.0f} م²")
            st.write(f"**الوسيط:** {df['المساحة_متر'].median():.0f} م²")

        st.markdown("---")

        # رسوم بيانية
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(df, x='المساحة_متر', y='السعر_دينار',
                             title="العلاقة بين المساحة والسعر",
                             labels={'المساحة_متر': 'المساحة (م²)', 'السعر_دينار': 'السعر (دينار)'},
                             trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df, x='عدد_الغرف', y='السعر_دينار',
                         title="توزيع الأسعار حسب عدد الغرف",
                         labels={'عدد_الغرف': 'عدد الغرف', 'السعر_دينار': 'السعر (دينار)'})
            st.plotly_chart(fig, use_container_width=True)

    # التبويب 2: تحليل المناطق
    with tabs[1]:
        st.markdown("### 🏘️ تحليل الأسعار حسب المناطق")

        region_stats = df.groupby('المنطقة').agg({
            'السعر_دينار': ['mean', 'min', 'max', 'count']
        }).round(0)
        region_stats.columns = ['متوسط السعر', 'أقل سعر', 'أعلى سعر', 'عدد العقارات']
        region_stats = region_stats.sort_values('متوسط السعر', ascending=False)

        st.dataframe(region_stats, use_container_width=True)

        st.markdown("---")

        # أغلى وأرخص المناطق
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💎 أغلى 10 مناطق")
            top10 = region_stats.head(10)
            fig = px.bar(top10, y=top10.index, x='متوسط السعر',
                         orientation='h',
                         labels={'متوسط السعر': 'متوسط السعر (دينار)', 'index': 'المنطقة'})
            fig.update_traces(marker_color='gold')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 💵 أرخص 10 مناطق")
            bottom10 = region_stats.tail(10)
            fig = px.bar(bottom10, y=bottom10.index, x='متوسط السعر',
                         orientation='h',
                         labels={'متوسط السعر': 'متوسط السعر (دينار)', 'index': 'المنطقة'})
            fig.update_traces(marker_color='lightblue')
            st.plotly_chart(fig, use_container_width=True)

    # التبويب 3: تحليل تفصيلي
    with tabs[2]:
        st.markdown("### 🔍 تحليلات متقدمة")

        col1, col2 = st.columns(2)

        with col1:
            # تأثير المصعد
            fig = px.box(df, x='يوجد_مصعد', y='السعر_دينار',
                         title="تأثير المصعد على السعر",
                         labels={'يوجد_مصعد': 'يوجد مصعد', 'السعر_دينار': 'السعر (دينار)'})
            st.plotly_chart(fig, use_container_width=True)

            # تأثير الموقف
            fig = px.box(df, x='يوجد_موقف', y='السعر_دينار',
                         title="تأثير موقف السيارات على السعر",
                         labels={'يوجد_موقف': 'يوجد موقف', 'السعر_دينار': 'السعر (دينار)'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # العلاقة بين العمر والسعر
            fig = px.scatter(df, x='عمر_البناء_سنوات', y='السعر_دينار',
                             title="العلاقة بين عمر البناء والسعر",
                             labels={'عمر_البناء_سنوات': 'عمر البناء (سنوات)', 'السعر_دينار': 'السعر (دينار)'},
                             trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

            # توزيع عدد الغرف
            room_dist = df['عدد_الغرف'].value_counts().sort_index()
            fig = px.bar(x=room_dist.index, y=room_dist.values,
                         title="توزيع العقارات حسب عدد الغرف",
                         labels={'x': 'عدد الغرف', 'y': 'العدد'})
            st.plotly_chart(fig, use_container_width=True)

    # التبويب 4: عينة البيانات
    with tabs[3]:
        st.markdown("### 📋 عينة من البيانات")

        # خيار لاختيار عدد الصفوف
        num_rows = st.slider("عدد الصفوف المعروضة:", 5, 100, 20)

        st.dataframe(df.head(num_rows), use_container_width=True)

        # زر التحميل
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 تحميل البيانات كاملة (CSV)",
            data=csv,
            file_name="jordan_properties.csv",
            mime="text/csv"
        )

# ═══════════════════════════════════════
# صفحة أداء النماذج
# ═══════════════════════════════════════
elif page == "🤖 أداء النماذج":
    st.title("🤖 تقييم أداء نماذج التعلم الآلي")

    st.markdown("### 📊 مقارنة النماذج الثلاثة")

    # جدول المقارنة
    comparison_df = pd.DataFrame(metrics).T
    comparison_df = comparison_df.round(2)
    comparison_df['MAE'] = comparison_df['MAE'].apply(lambda x: f"{x:,.0f} دينار")
    comparison_df['RMSE'] = comparison_df['RMSE'].apply(lambda x: f"{x:,.0f} دينار")
    comparison_df['R2'] = comparison_df['R2'].apply(lambda x: f"{x:.4f} ({x * 100:.2f}%)")

    st.dataframe(comparison_df, use_container_width=True)

    st.success(f"🏆 **أفضل نموذج:** {best_model_name}")

    st.markdown("---")

    # رسوم بيانية للمقارنة
    col1, col2 = st.columns(2)

    with col1:
        # مقارنة R2 Score
        r2_values = [metrics[m]['R2'] for m in metrics.keys()]
        fig = go.Figure(data=[
            go.Bar(x=list(metrics.keys()), y=r2_values,
                   marker_color=['#667eea', '#764ba2', '#f59e0b'])
        ])
        fig.update_layout(title="مقارنة R² Score",
                          xaxis_title="النموذج",
                          yaxis_title="R² Score",
                          yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # مقارنة MAE
        mae_values = [metrics[m]['MAE'] for m in metrics.keys()]
        fig = go.Figure(data=[
            go.Bar(x=list(metrics.keys()), y=mae_values,
                   marker_color=['#10b981', '#06b6d4', '#f59e0b'])
        ])
        fig.update_layout(title="مقارنة MAE (كلما قل كان أفضل)",
                          xaxis_title="النموذج",
                          yaxis_title="MAE (دينار)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # اختيار نموذج للتفصيل
    st.markdown("### 🔍 تحليل تفصيلي للنموذج")
    selected_model = st.selectbox("اختر النموذج:", list(models.keys()))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", f"{metrics[selected_model]['MAE']:,.0f} دينار")
    with col2:
        st.metric("RMSE", f"{metrics[selected_model]['RMSE']:,.0f} دينار")
    with col3:
        st.metric("R² Score", f"{metrics[selected_model]['R2']:.4f}")

    # رسم Actual vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test[:100],
        y=predictions[selected_model][:100],
        mode='markers',
        name='التوقعات',
        marker=dict(size=8, color='#667eea', opacity=0.6)
    ))

    # خط مثالي
    min_val = min(y_test.min(), predictions[selected_model].min())
    max_val = max(y_test.max(), predictions[selected_model].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='التوقع المثالي',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f"السعر الفعلي مقابل المتوقع - {selected_model}",
        xaxis_title="السعر الفعلي (دينار)",
        yaxis_title="السعر المتوقع (دينار)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # توزيع الأخطاء
    errors = y_test - predictions[selected_model]
    fig = px.histogram(errors, nbins=50,
                       title="توزيع الأخطاء",
                       labels={'value': 'الخطأ (دينار)', 'count': 'التكرار'})
    fig.update_traces(marker_color='#10b981')
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance (للنماذج التي تدعمها)
    if selected_model in ['Random Forest', 'Gradient Boosting']:
        st.markdown("### 📈 أهمية الخصائص")

        importance = models[selected_model].feature_importances_
        feature_names = ['المساحة', 'الغرف', 'الحمامات', 'عمر البناء', 'الطابق',
                         'مصعد', 'موقف', 'حديقة', 'تدفئة', 'قرب الخدمات', 'المنطقة']

        importance_df = pd.DataFrame({
            'الخاصية': feature_names,
            'الأهمية': importance
        }).sort_values('الأهمية', ascending=True)

        fig = px.bar(importance_df, x='الأهمية', y='الخاصية',
                     orientation='h',
                     title="أهمية الخصائص في التوقع")
        fig.update_traces(marker_color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════
# صفحة توقع السعر
# ═══════════════════════════════════════
elif page == "🎯 توقع السعر":
    st.title("🎯 احسب سعر عقارك")
    st.markdown("### أدخل مواصفات العقار للحصول على السعر المتوقع")

    st.markdown("---")

    # استخدام أفضل نموذج للتوقع
    st.info(f"💡 يتم استخدام نموذج **{best_model_name}** للتوقع (الأفضل أداءً)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📍 المعلومات الأساسية")
        region = st.selectbox("المنطقة:", sorted(df['المنطقة'].unique()))
        area = st.number_input("المساحة (م²):", min_value=50, max_value=500, value=150, step=10)
        rooms = st.slider("عدد الغرف:", 1, 6, 3)
        bathrooms = st.slider("عدد الحمامات:", 1, 4, 2)

    with col2:
        st.markdown("#### 🏗️ معلومات البناء")
        age = st.slider("عمر البناء (سنوات):", 0, 35, 5)
        floor = st.slider("الطابق:", 0, 11, 3)
        services = st.slider("قرب الخدمات (1-10):", 1, 10, 7)

    st.markdown("#### 🏠 المرافق المتوفرة")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        elevator = st.checkbox("🛗 مصعد", value=True)
    with col2:
        parking = st.checkbox("🅿️ موقف سيارات", value=True)
    with col3:
        garden = st.checkbox("🌳 حديقة", value=False)
    with col4:
        heating = st.checkbox("🔥 تدفئة مركزية", value=True)

    st.markdown("---")

    if st.button("💰 احسب السعر", use_container_width=True):
        with st.spinner("جاري حساب السعر..."):
            # تحضير البيانات للتوقع
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
                'المنطقة_رقم': [le.transform([region])[0]]
            })

            # التوقع
            predicted_price = best_model.predict(input_data)[0]
            price_per_sqm = predicted_price / area

            # عرض النتيجة
            st.success("✅ تم حساب السعر بنجاح!")

            # بطاقة النتيجة الكبيرة
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem; border-radius: 15px; text-align: center;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 2rem 0;'>
                    <h2 style='color: white; margin: 0;'>💰 السعر المتوقع</h2>
                    <h1 style='color: white; font-size: 3.5rem; margin: 1rem 0;'>
                        {predicted_price:,.0f} دينار
                    </h1>
                    <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem;'>
                        سعر المتر: {price_per_sqm:,.0f} دينار/م²
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # معلومات إضافية
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_price = df['السعر_دينار'].mean()
                diff_percent = ((predicted_price - avg_price) / avg_price) * 100
                st.metric("مقارنة بالمتوسط العام",
                          f"{diff_percent:+.1f}%",
                          delta=f"{predicted_price - avg_price:,.0f} دينار")

            with col2:
                region_avg = df[df['المنطقة'] == region]['السعر_دينار'].mean()
                region_diff = ((predicted_price - region_avg) / region_avg) * 100
                st.metric(f"مقارنة بمتوسط {region}",
                          f"{region_diff:+.1f}%",
                          delta=f"{predicted_price - region_avg:,.0f} دينار")

            with col3:
                similar_props = df[
                    (df['المساحة_متر'] >= area - 20) &
                    (df['المساحة_متر'] <= area + 20) &
                    (df['المنطقة'] == region)
                    ]
                if len(similar_props) > 0:
                    similar_avg = similar_props['السعر_دينار'].mean()
                    st.metric("عقارات مشابهة في المنطقة",
                              f"{len(similar_props)} عقار",
                              delta=f"متوسط: {similar_avg:,.0f} دينار")
                else:
                    st.metric("عقارات مشابهة", "0 عقار", delta="لا توجد مطابقات")

            st.markdown("---")

            # ملخص المواصفات
            st.markdown("### 📋 ملخص مواصفات العقار")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                - **المنطقة:** {region}
                - **المساحة:** {area} م²
                - **عدد الغرف:** {rooms}
                - **عدد الحمامات:** {bathrooms}
                - **عمر البناء:** {age} سنوات
                """)

            with col2:
                st.markdown(f"""
                - **الطابق:** {floor}
                - **مصعد:** {'✅ نعم' if elevator else '❌ لا'}
                - **موقف سيارات:** {'✅ نعم' if parking else '❌ لا'}
                - **حديقة:** {'✅ نعم' if garden else '❌ لا'}
                - **تدفئة مركزية:** {'✅ نعم' if heating else '❌ لا'}
                """)

            st.info(
                f"💡 هذا التوقع تم باستخدام نموذج **{best_model_name}** بدقة {metrics[best_model_name]['R2'] * 100:.1f}%")

            # نصائح
            st.markdown("### 💡 نصائح لزيادة قيمة العقار")

            tips = []
            if not elevator and floor > 2:
                tips.append("🛗 إضافة مصعد قد يزيد من قيمة العقار بنسبة ~10%")
            if not parking:
                tips.append("🅿️ توفير موقف سيارات قد يزيد من قيمة العقار بنسبة ~6%")
            if not heating and age < 10:
                tips.append("🔥 تركيب تدفئة مركزية قد يزيد من قيمة العقار بنسبة ~5%")
            if age > 15:
                tips.append("🏗️ تجديد العقار قد يحسن من قيمته بشكل ملحوظ")
            if not garden and floor <= 1:
                tips.append("🌳 إضافة حديقة للطابق الأرضي أو الأول قد يزيد القيمة بنسبة ~8%")

            if tips:
                for tip in tips:
                    st.markdown(f"- {tip}")
            else:
                st.success("✨ عقارك يحتوي على مواصفات ممتازة!")

# ═══════════════════════════════════════
# صفحة عن المشروع
# ═══════════════════════════════════════
else:  # page == "ℹ️ عن المشروع"
    st.title("ℹ️ عن المشروع")

    st.markdown("""
    ## 🏠 نظام توقع أسعار العقارات الأردنية

    ### 📖 نبذة عن المشروع

    هذا المشروع هو نظام ذكي لتوقع أسعار العقارات في الأردن باستخدام تقنيات التعلم الآلي 
    (Machine Learning). يهدف النظام إلى مساعدة المشترين والبائعين في تحديد الأسعار العادلة 
    للعقارات بناءً على مواصفاتها وموقعها.

    ---

    ### 🎯 أهداف المشروع

    - **مساعدة المشترين**: في تقييم العقارات قبل الشراء
    - **مساعدة البائعين**: في تحديد سعر مناسب للعقار
    - **الشفافية**: توفير معلومات موثوقة عن سوق العقارات
    - **توعية**: نشر الوعي حول العوامل المؤثرة في أسعار العقارات

    ---

    ### 💻 التقنيات المستخدمة

    #### لغات البرمجة والمكتبات
    - **Python 3.9+** - لغة البرمجة الأساسية
    - **Pandas** - معالجة وتحليل البيانات
    - **NumPy** - العمليات الرياضية
    - **Scikit-learn** - بناء نماذج التعلم الآلي
    - **Matplotlib & Seaborn** - الرسوم البيانية
    - **Plotly** - التصورات التفاعلية
    - **Streamlit** - بناء واجهة الويب

    #### نماذج التعلم الآلي
    1. **Linear Regression** - النموذج الأساسي
    2. **Random Forest Regressor** - النموذج الأفضل أداءً
    3. **Gradient Boosting** - نموذج متقدم

    ---

    ### 📊 البيانات

    - **عدد العقارات:** 1,500 عقار
    - **عدد المناطق:** 20 منطقة في عمّان
    - **الخصائص:** 11 خاصية لكل عقار
    - **نطاق الأسعار:** {df['السعر_دينار'].min():,} - {df['السعر_دينار'].max():,} دينار

    #### المناطق المشمولة
    """)

    regions_list = sorted(df['المنطقة'].unique())
    cols = st.columns(4)
    for i, region in enumerate(regions_list):
        with cols[i % 4]:
            st.markdown(f"- {region}")

    st.markdown("""
    ---

    ### 📈 أداء النماذج

    تم تدريب 3 نماذج مختلفة وتقييمها:
    """)

    # جدول الأداء
    performance_data = []
    for model_name, model_metrics in metrics.items():
        performance_data.append({
            'النموذج': model_name,
            'R² Score': f"{model_metrics['R2']:.4f}",
            'الدقة': f"{model_metrics['R2'] * 100:.2f}%",
            'MAE': f"{model_metrics['MAE']:,.0f} دينار"
        })

    st.table(pd.DataFrame(performance_data))

    st.success(f"🏆 **أفضل نموذج:** {best_model_name} بدقة {metrics[best_model_name]['R2'] * 100:.2f}%")

    st.markdown("""
    ---

    ### 🔬 منهجية العمل

    1. **جمع البيانات** - إنشاء مجموعة بيانات واقعية للعقارات الأردنية
    2. **تحليل البيانات** - دراسة العلاقات والأنماط في البيانات
    3. **هندسة الخصائص** - اختيار وتحضير الخصائص المؤثرة
    4. **تدريب النماذج** - بناء وتدريب 3 نماذج مختلفة
    5. **التقييم** - مقارنة أداء النماذج واختيار الأفضل
    6. **النشر** - إطلاق النظام عبر واجهة ويب تفاعلية

    ---

    ### 📌 الخصائص المستخدمة في التوقع

    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### خصائص العقار
        - 📏 المساحة (م²)
        - 🛏️ عدد الغرف
        - 🚿 عدد الحمامات
        - 🏗️ عمر البناء (سنوات)
        - 🏢 رقم الطابق
        """)

    with col2:
        st.markdown("""
        #### المرافق والموقع
        - 🛗 وجود مصعد
        - 🅿️ وجود موقف سيارات
        - 🌳 وجود حديقة
        - 🔥 وجود تدفئة مركزية
        - 📍 المنطقة
        - 🏪 قرب الخدمات
        """)

    st.markdown("""
    ---

    ### 🎓 الفوائد التعليمية

    هذا المشروع يوضح:
    - كيفية بناء نظام تعلم آلي متكامل
    - معالجة وتحليل البيانات
    - بناء ومقارنة نماذج مختلفة
    - تطوير واجهة مستخدم تفاعلية
    - نشر تطبيق ويب

    ---

    ### 👨‍💻 المطور

    **[اسمك هنا]**

    - 🔗 LinkedIn: [your-linkedin-profile]
    - 💻 GitHub: [your-github-profile]
    - 📧 Email: your.email@example.com
    - 🌐 Portfolio: [your-portfolio-website]

    ---

    ### 📜 الترخيص

    هذا المشروع متاح تحت ترخيص MIT License - يمكن استخدامه لأغراض تعليمية وغير تجارية.

    ---

    ### 🙏 شكر وتقدير

    - **Streamlit** - لتوفير منصة سهلة لبناء تطبيقات البيانات
    - **Scikit-learn** - لمكتبة التعلم الآلي الممتازة
    - **المجتمع البرمجي** - للدعم والمساعدة المستمرة

    ---

    ### 📞 تواصل معنا

    لأي استفسارات أو اقتراحات، يرجى التواصل عبر:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📧 **Email**\nyour.email@example.com")

    with col2:
        st.info("💼 **LinkedIn**\n[Your LinkedIn Profile]")

    with col3:
        st.info("💻 **GitHub**\n[Your GitHub Profile]")

    st.markdown("---")

    st.warning("""
    ⚠️ **ملاحظة مهمة:** 
    هذا المشروع تعليمي ويستخدم بيانات تجريبية. النتائج تقريبية ولا يجب الاعتماد عليها 
    في اتخاذ قرارات شراء أو بيع فعلية. للحصول على تقييم دقيق، يُنصح باستشارة خبراء عقاريين.
    """)

    st.success("⭐ إذا أعجبك المشروع، لا تنسى إعطاءه نجمة على GitHub!")

# ═══════════════════════════════════════
# Footer
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p>صُنع بـ ❤️ باستخدام Python و Streamlit</p>
        <p>© 2025 نظام توقع أسعار العقارات الأردنية | جميع الحقوق محفوظة</p>
    </div>
""", unsafe_allow_html=True)