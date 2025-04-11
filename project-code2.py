import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load data
file_path = r"C:\Users\akhil\OneDrive\Desktop\INT-375\project\PROJECT.csv"
df = pd.read_csv(file_path)

print(df.info())
print(df.describe())

#  Handle Missing Values 
df['pollutant_id'].fillna(df['pollutant_id'].mode()[0], inplace=True)
df['pollutant_avg'].fillna(df['pollutant_avg'].mean(), inplace=True)
df['pollutant_min'].fillna(df['pollutant_min'].mean(), inplace=True)
df['pollutant_max'].fillna(df['pollutant_max'].mean(), inplace=True)
df['station'].fillna(df['station'].mode()[0], inplace=True)
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')


#Objective 1: Statewise Average Pollution Levels (Bar Chart)
state_avg = df.groupby('state')['pollutant_avg'].mean().sort_values()

# Plot horizontal bar chart
plt.figure(figsize=(10, 8))
state_avg.plot(kind='barh', color='skyblue')
plt.title('Average Pollution Levels by State')
plt.xlabel('Average Pollutant Level')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# Objective 2: Top and Bottom Cities by Pollution Level (Vertical Bar Chart)
city_avg = df.groupby('city')['pollutant_avg'].mean().sort_values()
top_10 = city_avg.tail(10)
bottom_10 = city_avg.head(10)
combined = pd.concat([bottom_10, top_10])
plt.figure(figsize=(14, 7))
bars = plt.bar(combined.index, combined.values, 
               color=['#1f77b4']*10 + ['#ff7f0e']*10, edgecolor='black')
plt.title('Top 10 and Bottom 10 Cities by Average Pollution Level', fontsize=16, weight='bold', pad=20)
plt.xlabel('City', fontsize=14)
plt.ylabel('Average Pollutant Level', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)
plt.legend(['Bottom 10 (Low Pollution)', 'Top 10 (High Pollution)'], fontsize=12, loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Objective 3: Pollution Intensity Map (Geographical Scatter Plot)
geo_data = df.groupby(['city', 'latitude', 'longitude'])['pollutant_avg'].mean().reset_index()
plt.figure(figsize=(12, 8))
scatter = plt.scatter(geo_data['longitude'], geo_data['latitude'], 
                     s=geo_data['pollutant_avg']*20, c=geo_data['pollutant_avg'], 
                     cmap='crest', alpha=0.7, edgecolor='black')
plt.colorbar(scatter, label='Average Pollutant Level', shrink=0.8)
plt.title('Pollution Intensity Across Cities', fontsize=16, weight='bold', pad=20)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.gca().set_facecolor('#f5f5f5')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Objective 4: Pollutant Distribution by Type (Donut Chart)
pollutant_counts = df['pollutant_id'].value_counts()
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(pollutant_counts, labels=pollutant_counts.index, 
                                   autopct='%1.1f%%', startangle=90, 
                                   colors=sb.color_palette('Set2'), 
                                   pctdistance=0.85, textprops={'fontsize': 12})
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.title('Distribution of Pollutant Types', fontsize=16, weight='bold', pad=20)
total_readings = pollutant_counts.sum()
plt.text(0, 0, f'Total\n{total_readings}', ha='center', va='center', fontsize=14, weight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Objective 5: Predictive Analysis with Regression Plot
# Prepare features and target
X = df[['pollutant_min', 'pollutant_max']]
y = df['pollutant_avg']

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Calculate R² score
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.4f}')

# Plot scatter with regression line for pollutant_max
plt.figure(figsize=(10, 6))
sb.scatterplot(x=df['pollutant_max'], y=df['pollutant_avg'], alpha=0.5, color='#FF7F50')
# Compute regression line for pollutant_max, holding pollutant_min at mean
slope = model.coef_[1]
intercept = model.intercept_ - model.coef_[0] * df['pollutant_min'].mean()
x_range = np.array([df['pollutant_max'].min(), df['pollutant_max'].max()])
y_range = slope * x_range + intercept
plt.plot(x_range, y_range, color='red', label='Regression Line')
# Add city name annotations for top 5 pollutant_avg values
top_cities = df.nlargest(5, 'pollutant_avg')
for i, row in top_cities.iterrows():
    plt.annotate(row['city'], (row['pollutant_max'], row['pollutant_avg']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=10)
plt.title('Regression Plot: Pollutant Level vs Maximum Levels')
plt.xlabel('Maximum Pollutant Level')
plt.ylabel('Average Pollutant Level')
plt.legend()
plt.tight_layout()

# Save the plot for LinkedIn
plt.savefig('pollution_regression.png', dpi=300, bbox_inches='tight')
plt.show()