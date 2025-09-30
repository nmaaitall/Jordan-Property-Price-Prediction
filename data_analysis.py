import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# إعدادات لدعم العربية في الرسوم البيانية
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# قراءة البيانات
print("📂 جاري قراءة البيانات...")
df = pd.read_csv('jordan_properties.csv')

print("✅ تم تحميل البيانات بنجاح!\n")

# ═══════════════════════════════════════
# 1️⃣ نظرة عامة على البيانات
# ═══════════════════════════════════════
print("="*60)
print("📊 نظرة عامة على البيانات")
print("="*60)
print(f"عدد العقارات: {len(df)}")
print(f"عدد الخصائص: {df.shape[1]}")
print(f"\nأسماء الأعمدة:")
print(df.columns.tolist())

# ═══════════════════════════════════════
# 2️⃣ الإحصائيات الوصفية
# ═══════════════════════════════════════
print("\n" + "="*60)
print("📈 الإحصائيات الوصفية")
print("="*60)
print(df.describe())

# ═══════════════════════════════════════
# 3️⃣ التحقق من القيم المفقودة
# ═══════════════════════════════════════
print("\n" + "="*60)
print("🔍 التحقق من القيم المفقودة")
print("="*60)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ لا توجد قيم مفقودة - البيانات نظيفة!")
else:
    print(missing[missing > 0])

# ═══════════════════════════════════════
# 4️⃣ معلومات عن الأسعار
# ═══════════════════════════════════════
print("\n" + "="*60)
print("💰 تحليل الأسعار")
print("="*60)
print(f"أقل سعر: {df['السعر_دينار'].min():,} دينار")
print(f"أعلى سعر: {df['السعر_دينار'].max():,} دينار")
print(f"متوسط السعر: {df['السعر_دينار'].mean():,.0f} دينار")
print(f"الوسيط: {df['السعر_دينار'].median():,.0f} دينار")

# ═══════════════════════════════════════
# 5️⃣ متوسط الأسعار حسب المنطقة
# ═══════════════════════════════════════
print("\n" + "="*60)
print("🏘️ متوسط الأسعار حسب المنطقة (أغلى 10 مناطق)")
print("="*60)
avg_price_by_region = df.groupby('المنطقة')['السعر_دينار'].mean().sort_values(ascending=False)
print(avg_price_by_region.head(10).apply(lambda x: f"{x:,.0f} دينار"))

# ═══════════════════════════════════════
# 6️⃣ الرسوم البيانية
# ═══════════════════════════════════════
print("\n" + "="*60)
print("📊 جاري إنشاء الرسوم البيانية...")
print("="*60)

# إنشاء figure كبيرة
fig = plt.figure(figsize=(16, 12))

# 1. توزيع الأسعار
plt.subplot(3, 3, 1)
plt.hist(df['السعر_دينار'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Prices', fontsize=12, fontweight='bold')
plt.xlabel('Price (JOD)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 2. توزيع المساحات
plt.subplot(3, 3, 2)
plt.hist(df['المساحة_متر'], bins=40, color='lightgreen', edgecolor='black')
plt.title('Distribution of Areas', fontsize=12, fontweight='bold')
plt.xlabel('Area (sqm)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 3. العلاقة بين المساحة والسعر
plt.subplot(3, 3, 3)
plt.scatter(df['المساحة_متر'], df['السعر_دينار'], alpha=0.5, color='coral')
plt.title('Price vs Area', fontsize=12, fontweight='bold')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (JOD)')
plt.grid(alpha=0.3)

# 4. توزيع عدد الغرف
plt.subplot(3, 3, 4)
df['عدد_الغرف'].value_counts().sort_index().plot(kind='bar', color='gold', edgecolor='black')
plt.title('Number of Rooms Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Number of Rooms')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

# 5. العلاقة بين عدد الغرف والسعر
plt.subplot(3, 3, 5)
df.groupby('عدد_الغرف')['السعر_دينار'].mean().plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Avg Price by Number of Rooms', fontsize=12, fontweight='bold')
plt.xlabel('Number of Rooms')
plt.ylabel('Avg Price (JOD)')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

# 6. أغلى 10 مناطق
plt.subplot(3, 3, 6)
top_10 = avg_price_by_region.head(10)
plt.barh(range(len(top_10)), top_10.values, color='purple', alpha=0.7)
plt.yticks(range(len(top_10)), top_10.index, fontsize=9)
plt.title('Top 10 Most Expensive Areas', fontsize=12, fontweight='bold')
plt.xlabel('Avg Price (JOD)')
plt.grid(alpha=0.3)

# 7. تأثير المصعد على السعر
plt.subplot(3, 3, 7)
df.groupby('يوجد_مصعد')['السعر_دينار'].mean().plot(kind='bar', color=['salmon', 'lightgreen'], edgecolor='black')
plt.title('Impact of Elevator on Price', fontsize=12, fontweight='bold')
plt.xlabel('Has Elevator (0=No, 1=Yes)')
plt.ylabel('Avg Price (JOD)')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

# 8. توزيع عمر البناء
plt.subplot(3, 3, 8)
plt.hist(df['عمر_البناء_سنوات'], bins=30, color='orange', edgecolor='black')
plt.title('Building Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 9. العلاقة بين عمر البناء والسعر
plt.subplot(3, 3, 9)
plt.scatter(df['عمر_البناء_سنوات'], df['السعر_دينار'], alpha=0.5, color='teal')
plt.title('Price vs Building Age', fontsize=12, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Price (JOD)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('jordan_properties_analysis.png', dpi=300, bbox_inches='tight')
print("✅ تم حفظ الرسوم البيانية في ملف: jordan_properties_analysis.png")
plt.show()

print("\n" + "="*60)
print("✅ انتهى التحليل بنجاح!")
print("="*60)