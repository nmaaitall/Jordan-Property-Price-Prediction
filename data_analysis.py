import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib to support Arabic text in visualizations
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# Load the dataset
print("Loading data from CSV file...")
df = pd.read_csv('jordan_properties.csv')

print("Data loaded successfully!\n")

# ═══════════════════════════════════════
# Data Overview
# ═══════════════════════════════════════
print("="*60)
print("Dataset Overview")
print("="*60)
print(f"Total properties: {len(df)}")
print(f"Number of features: {df.shape[1]}")
print(f"\nColumn names:")
print(df.columns.tolist())

# ═══════════════════════════════════════
# Descriptive Statistics
# ═══════════════════════════════════════
print("\n" + "="*60)
print("Descriptive Statistics")
print("="*60)
print(df.describe())

# ═══════════════════════════════════════
# Missing Values Check
# ═══════════════════════════════════════
print("\n" + "="*60)
print("Missing Values Analysis")
print("="*60)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values detected - dataset is clean")
else:
    print(missing[missing > 0])

# ═══════════════════════════════════════
# Price Analysis
# ═══════════════════════════════════════
print("\n" + "="*60)
print("Price Analysis")
print("="*60)
print(f"Minimum price: {df['السعر_دينار'].min():,} JOD")
print(f"Maximum price: {df['السعر_دينار'].max():,} JOD")
print(f"Average price: {df['السعر_دينار'].mean():,.0f} JOD")
print(f"Median price: {df['السعر_دينار'].median():,.0f} JOD")

# ═══════════════════════════════════════
# Average Prices by Region
# ═══════════════════════════════════════
print("\n" + "="*60)
print("Average Prices by Region (Top 10 Most Expensive)")
print("="*60)
avg_price_by_region = df.groupby('المنطقة')['السعر_دينار'].mean().sort_values(ascending=False)
print(avg_price_by_region.head(10).apply(lambda x: f"{x:,.0f} JOD"))

# ═══════════════════════════════════════
# Data Visualizations
# ═══════════════════════════════════════
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Price distribution histogram
plt.subplot(3, 3, 1)
plt.hist(df['السعر_دينار'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Prices', fontsize=12, fontweight='bold')
plt.xlabel('Price (JOD)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 2. Area distribution histogram
plt.subplot(3, 3, 2)
plt.hist(df['المساحة_متر'], bins=40, color='lightgreen', edgecolor='black')
plt.title('Distribution of Areas', fontsize=12, fontweight='bold')
plt.xlabel('Area (sqm)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 3. Scatter plot: Price vs Area
plt.subplot(3, 3, 3)
plt.scatter(df['المساحة_متر'], df['السعر_دينار'], alpha=0.5, color='coral')
plt.title('Price vs Area', fontsize=12, fontweight='bold')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (JOD)')
plt.grid(alpha=0.3)

# 4. Number of rooms distribution
plt.subplot(3, 3, 4)
df['عدد_الغرف'].value_counts().sort_index().plot(kind='bar', color='gold', edgecolor='black')
plt.title('Number of Rooms Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Number of Rooms')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

# 5. Average price by number of rooms
plt.subplot(3, 3, 5)
df.groupby('عدد_الغرف')['السعر_دينار'].mean().plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Avg Price by Number of Rooms', fontsize=12, fontweight='bold')
plt.xlabel('Number of Rooms')
plt.ylabel('Avg Price (JOD)')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

# 6. Top 10 most expensive areas
plt.subplot(3, 3, 6)
top_10 = avg_price_by_region.head(10)
plt.barh(range(len(top_10)), top_10.values, color='purple', alpha=0.7)
plt.yticks(range(len(top_10)), top_10.index, fontsize=9)
plt.title('Top 10 Most Expensive Areas', fontsize=12, fontweight='bold')
plt.xlabel('Avg Price (JOD)')
plt.grid(alpha=0.3)

# 7. Impact of elevator on price
plt.subplot(3, 3, 7)
df.groupby('يوجد_مصعد')['السعر_دينار'].mean().plot(kind='bar', color=['salmon', 'lightgreen'], edgecolor='black')
plt.title('Impact of Elevator on Price', fontsize=12, fontweight='bold')
plt.xlabel('Has Elevator (0=No, 1=Yes)')
plt.ylabel('Avg Price (JOD)')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

# 8. Building age distribution
plt.subplot(3, 3, 8)
plt.hist(df['عمر_البناء_سنوات'], bins=30, color='orange', edgecolor='black')
plt.title('Building Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 9. Price vs building age correlation
plt.subplot(3, 3, 9)
plt.scatter(df['عمر_البناء_سنوات'], df['السعر_دينار'], alpha=0.5, color='teal')
plt.title('Price vs Building Age', fontsize=12, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Price (JOD)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('jordan_properties_analysis.png', dpi=300, bbox_inches='tight')
print("Visualizations saved to: jordan_properties_analysis.png")
plt.show()

print("\n" + "="*60)
print("Analysis completed successfully")
print("="*60)