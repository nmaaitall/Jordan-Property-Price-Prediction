import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Jordanian regions with average price per square meter (2024-2025)
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

# Number of properties to generate
n_properties = 1500

# Initialize data dictionary
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
    'قرب_الخدمات': [],  # Scale: 1-10
    'السعر_دينار': []
}

for _ in range(n_properties):
    # Select random region
    region = random.choice(list(regions.keys()))
    base_price = regions[region]

    # Generate realistic property specifications
    area = np.random.randint(80, 400)

    # Number of rooms based on area (more realistic distribution)
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

    # Elevator more likely in higher floors and expensive areas
    has_elevator = 1 if (floor > 2 and random.random() > 0.25) or (floor > 4) else 0

    has_parking = 1 if random.random() > 0.25 else 0
    has_garden = 1 if floor <= 1 and random.random() > 0.75 else 0
    has_heating = 1 if base_price > 700 and random.random() > 0.4 else 0
    services_proximity = np.random.randint(1, 11)

    # Calculate realistic price
    price = base_price * area

    # Price adjustments based on property features
    price *= (1 - age * 0.012)  # Each year decreases value by 1.2%
    price *= (1 + rooms * 0.04)  # Each room adds 4%
    price *= (1 + has_elevator * 0.10)  # Elevator adds 10%
    price *= (1 + has_parking * 0.06)  # Parking adds 6%
    price *= (1 + has_garden * 0.08)  # Garden adds 8%
    price *= (1 + has_heating * 0.05)  # Central heating adds 5%
    price *= (1 + services_proximity * 0.015)  # Proximity to services

    # Floor-based adjustment
    if floor == 0:
        price *= 0.95  # Ground floor slightly cheaper
    elif floor >= 8:
        price *= 1.05  # Higher floors more expensive

    # Add realistic variance
    price *= np.random.uniform(0.88, 1.12)

    # Append data to dictionary
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

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('jordan_properties.csv', index=False, encoding='utf-8-sig')

print("Dataset created successfully!")
print(f"Total properties: {len(df)}")
print(f"\nSample data:")
print(df.head(10))
print(f"\nPrice range: {df['السعر_دينار'].min():,} - {df['السعر_دينار'].max():,} JOD")
print(f"Average price: {df['السعر_دينار'].mean():,.0f} JOD")
print(f"\nProperties distribution by region:")
print(df['المنطقة'].value_counts())