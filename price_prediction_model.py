import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("🤖 نظام توقع أسعار العقارات في الأردن - Machine Learning")
print("=" * 70)

# ═══════════════════════════════════════
# 1️⃣ تحميل البيانات
# ═══════════════════════════════════════
print("\n📂 جاري تحميل البيانات...")
df = pd.read_csv('jordan_properties.csv')
print(f"✅ تم تحميل {len(df)} عقار بنجاح!\n")

# ═══════════════════════════════════════
# 2️⃣ تحضير البيانات للنموذج
# ═══════════════════════════════════════
print("🔧 جاري تحضير البيانات...")

# تحويل المنطقة من نص لأرقام
le = LabelEncoder()
df['المنطقة_رقم'] = le.fit_transform(df['المنطقة'])

# اختيار الخصائص (Features) والهدف (Target)
features = ['المساحة_متر', 'عدد_الغرف', 'عدد_الحمامات', 'عمر_البناء_سنوات',
            'طابق', 'يوجد_مصعد', 'يوجد_موقف', 'يوجد_حديقة',
            'يوجد_تدفئة_مركزية', 'قرب_الخدمات', 'المنطقة_رقم']

X = df[features]
y = df['السعر_دينار']

print(f"✅ عدد الخصائص: {len(features)}")
print(f"✅ الخصائص المستخدمة: {features}\n")

# ═══════════════════════════════════════
# 3️⃣ تقسيم البيانات
# ═══════════════════════════════════════
print("✂️ جاري تقسيم البيانات...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ بيانات التدريب: {len(X_train)} عقار")
print(f"✅ بيانات الاختبار: {len(X_test)} عقار\n")

# ═══════════════════════════════════════
# 4️⃣ بناء وتدريب النماذج
# ═══════════════════════════════════════
print("=" * 70)
print("🏗️ جاري بناء وتدريب النماذج...")
print("=" * 70)

print("\n1️⃣ Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("   ✅ تم التدريب")

print("\n2️⃣ Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("   ✅ تم التدريب")

print("\n3️⃣ Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("   ✅ تم التدريب")

# ═══════════════════════════════════════
# 5️⃣ تقييم النماذج
# ═══════════════════════════════════════
print("\n" + "=" * 70)
print("📊 تقييم أداء النماذج")
print("=" * 70)

models_results = []

for name, predictions in [('Linear Regression', lr_pred),
                          ('Random Forest', rf_pred),
                          ('Gradient Boosting', gb_pred)]:
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    models_results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2_Score': r2
    })

    print(f"\n🔹 {name}:")
    print(f"   MAE: {mae:,.0f} دينار")
    print(f"   RMSE: {rmse:,.0f} دينار")
    print(f"   R² Score: {r2:.4f} ({r2 * 100:.2f}%)")

results_df = pd.DataFrame(models_results)

# ═══════════════════════════════════════
# 6️⃣ اختيار أفضل نموذج
# ═══════════════════════════════════════
best_model_name = results_df.loc[results_df['R2_Score'].idxmax(), 'Model']
print("\n" + "=" * 70)
print(f"🏆 أفضل نموذج: {best_model_name}")
print("=" * 70)

if best_model_name == 'Linear Regression':
    best_model = lr_model
    best_pred = lr_pred
elif best_model_name == 'Random Forest':
    best_model = rf_model
    best_pred = rf_pred
else:
    best_model = gb_model
    best_pred = gb_pred

# ═══════════════════════════════════════
# 7️⃣ أهمية الخصائص
# ═══════════════════════════════════════
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n📈 أهمية الخصائص في التوقع:")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(feature_importance.to_string(index=False))

# ═══════════════════════════════════════
# 8️⃣ الرسوم البيانية
# ═══════════════════════════════════════
print("\n📊 جاري إنشاء الرسوم البيانية...")

fig = plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.bar(results_df['Model'], results_df['R2_Score'], color=['skyblue', 'lightgreen', 'coral'])
plt.title('Model Comparison - R² Score', fontweight='bold', fontsize=12)
plt.ylabel('R² Score')
plt.ylim([0, 1])
plt.xticks(rotation=15, ha='right')
plt.grid(alpha=0.3, axis='y')

plt.subplot(2, 3, 2)
plt.bar(results_df['Model'], results_df['MAE'], color=['gold', 'lightblue', 'pink'])
plt.title('Model Comparison - MAE', fontweight='bold', fontsize=12)
plt.ylabel('MAE (JOD)')
plt.xticks(rotation=15, ha='right')
plt.grid(alpha=0.3, axis='y')

plt.subplot(2, 3, 3)
plt.scatter(y_test, best_pred, alpha=0.6, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted - {best_model_name}', fontweight='bold', fontsize=12)
plt.xlabel('Actual Price (JOD)')
plt.ylabel('Predicted Price (JOD)')
plt.grid(alpha=0.3)

plt.subplot(2, 3, 4)
errors = y_test - best_pred
plt.hist(errors, bins=50, color='teal', edgecolor='black', alpha=0.7)
plt.title('Prediction Error Distribution', fontweight='bold', fontsize=12)
plt.xlabel('Error (JOD)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3, axis='y')

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    plt.subplot(2, 3, 5)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'], color='orange', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=9)
    plt.title('Top 10 Feature Importance', fontweight='bold', fontsize=12)
    plt.xlabel('Importance')
    plt.grid(alpha=0.3, axis='x')

plt.subplot(2, 3, 6)
percentage_error = np.abs((y_test - best_pred) / y_test) * 100
plt.hist(percentage_error, bins=50, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Percentage Error Distribution', fontweight='bold', fontsize=12)
plt.xlabel('Error (%)')
plt.ylabel('Frequency')
plt.axvline(percentage_error.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {percentage_error.mean():.1f}%')
plt.legend()
plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("✅ تم حفظ الرسوم البيانية في: model_evaluation.png")
plt.show()

# ═══════════════════════════════════════
# 9️⃣ اختبار النموذج
# ═══════════════════════════════════════
print("\n" + "=" * 70)
print("🧪 اختبار النموذج بأمثلة واقعية")
print("=" * 70)

example_1 = pd.DataFrame({
    'المساحة_متر': [150],
    'عدد_الغرف': [3],
    'عدد_الحمامات': [2],
    'عمر_البناء_سنوات': [5],
    'طابق': [3],
    'يوجد_مصعد': [1],
    'يوجد_موقف': [1],
    'يوجد_حديقة': [0],
    'يوجد_تدفئة_مركزية': [1],
    'قرب_الخدمات': [8],
    'المنطقة_رقم': [le.transform(['عبدون'])[0]]
})

predicted_price_1 = best_model.predict(example_1)[0]
print(f"\n🏠 مثال 1: شقة بعبدون (150م², 3 غرف، 5 سنوات)")
print(f"   💰 السعر المتوقع: {predicted_price_1:,.0f} دينار")

example_2 = pd.DataFrame({
    'المساحة_متر': [120],
    'عدد_الغرف': [2],
    'عدد_الحمامات': [1],
    'عمر_البناء_سنوات': [15],
    'طابق': [1],
    'يوجد_مصعد': [0],
    'يوجد_موقف': [0],
    'يوجد_حديقة': [0],
    'يوجد_تدفئة_مركزية': [0],
    'قرب_الخدمات': [5],
    'المنطقة_رقم': [le.transform(['ماركا'])[0]]
})

predicted_price_2 = best_model.predict(example_2)[0]
print(f"\n🏠 مثال 2: شقة بماركا (120م², 2 غرف، 15 سنة)")
print(f"   💰 السعر المتوقع: {predicted_price_2:,.0f} دينار")

print("\n" + "=" * 70)
print("✅ انتهى بناء النموذج بنجاح!")
print("=" * 70)