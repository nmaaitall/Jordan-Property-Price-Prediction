import pandas as pd
import numpy as np
import random

# تثبيت الـ random seed عشان نقدر نكرر النتائج
np.random.seed(42)
random.seed(42)

# المناطق الأردنية مع متوسط سعر المتر الحقيقي (2024-2025)
regions = {
    'عبدون': 1200,
    'دير غبار': 1000,
    'أم أذينة': 950,
    'الصويفية': 900,
    'خلدا': 850,
    'أم السماق': 820,
    'تلاع العلي': 800,
    'الجاردنز': 780,
    'الشميساني': 750,
    'اللويبدة': 700,
    'الجبيهة': 650,
    'صويلح': 600,
    'طبربور': 550,
    'ماركا': 500,
    'شفا بدران': 480,
    'الياسمين': 470,
    'المقابلين': 450,
    'الهاشمي الشمالي': 430,
    'جبل الحسين': 420,
    'النصر': 400,
}

# عدد العقارات
n_properties = 1500

# توليد البيانات
data = {
    'المنطقة': [],
    'المساحة_متر': [],
    'عدد_الغرف': [],
    'عدد_الحمامات': [],
    'عمر_البناء_سنوات': [],
    'طابق': [],
    'يوجد_مصعد': [],
    'يوجد_موقف': [],
    'يوجد_حديقة': [],
    'يوجد_تدفئة_مركزية': [],
    'قرب_الخدمات': [],  # 1-10
    'السعر_دينار': []
}

for _ in range(n_properties):
    # اختيار منطقة عشوائية
    region = random.choice(list(regions.keys()))
    base_price = regions[region]

    # توليد المواصفات بشكل واقعي
    area = np.random.randint(80, 400)

    # عدد الغرف حسب المساحة (أكثر واقعية)
    if area < 100:
        rooms = np.random.randint(1, 3)
    elif area < 150:
        rooms = np.random.randint(2, 4)
    elif area < 250:
        rooms = np.random.randint(3, 5)
    else:
        rooms = np.random.randint(4, 7)

    bathrooms = min(rooms, np.random.randint(1, 4))
    age = np.random.randint(0, 35)
    floor = np.random.randint(0, 12)

    # مصعد أكثر احتمالاً في الطوابق العالية والمناطق الغالية
    has_elevator = 1 if (floor > 2 and random.random() > 0.25) or (floor > 4) else 0

    has_parking = 1 if random.random() > 0.25 else 0
    has_garden = 1 if floor <= 1 and random.random() > 0.75 else 0
    has_heating = 1 if base_price > 700 and random.random() > 0.4 else 0
    services_proximity = np.random.randint(1, 11)

    # حساب السعر بطريقة واقعية
    price = base_price * area

    # تعديلات السعر
    price *= (1 - age * 0.012)  # كل سنة تقلل 1.2%
    price *= (1 + rooms * 0.04)  # كل غرفة تزيد 4%
    price *= (1 + has_elevator * 0.10)  # مصعد يزيد 10%
    price *= (1 + has_parking * 0.06)  # موقف يزيد 6%
    price *= (1 + has_garden * 0.08)  # حديقة تزيد 8%
    price *= (1 + has_heating * 0.05)  # تدفئة مركزية تزيد 5%
    price *= (1 + services_proximity * 0.015)  # قرب الخدمات

    # تعديل حسب الطابق
    if floor == 0:
        price *= 0.95  # الأرضي أقل شوي
    elif floor >= 8:
        price *= 1.05  # الطوابق العالية أغلى

    # إضافة تباين واقعي
    price *= np.random.uniform(0.88, 1.12)

    # إضافة البيانات
    data['المنطقة'].append(region)
    data['المساحة_متر'].append(area)
    data['عدد_الغرف'].append(rooms)
    data['عدد_الحمامات'].append(bathrooms)
    data['عمر_البناء_سنوات'].append(age)
    data['طابق'].append(floor)
    data['يوجد_مصعد'].append(has_elevator)
    data['يوجد_موقف'].append(has_parking)
    data['يوجد_حديقة'].append(has_garden)
    data['يوجد_تدفئة_مركزية'].append(has_heating)
    data['قرب_الخدمات'].append(services_proximity)
    data['السعر_دينار'].append(int(price))

# إنشاء DataFrame
df = pd.DataFrame(data)

# حفظ البيانات
df.to_csv('jordan_properties.csv', index=False, encoding='utf-8-sig')

print("✅ تم إنشاء dataset العقارات الأردنية بنجاح!")
print(f"📊 عدد العقارات: {len(df)}")
print(f"\n📋 عينة من البيانات:")
print(df.head(10))
print(f"\n💰 نطاق الأسعار: {df['السعر_دينار'].min():,} - {df['السعر_دينار'].max():,} دينار")
print(f"💰 متوسط السعر: {df['السعر_دينار'].mean():,.0f} دينار")
print(f"\n🏘️ توزيع العقارات حسب المنطقة:")
print(df['المنطقة'].value_counts())