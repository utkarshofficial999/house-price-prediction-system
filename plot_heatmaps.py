import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('house_prices.csv')

# 1. Full Correlation Matrix Heatmap
plt.figure(figsize=(14, 11))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr_matrix, 
    mask=mask, 
    cmap='coolwarm', 
    annot=True, 
    fmt='.2f', 
    linewidths=0.8, 
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"}, 
    vmin=-1, vmax=1
)
plt.title("Correlation Matrix Heatmap of Housing Features", fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=200)
plt.close()
print("[OK] Saved 'correlation_heatmap.png'")

# 2. Target Correlation Heatmap (Ranked Price Drivers)
plt.figure(figsize=(6, 10))
price_corr = corr_matrix[['Price']].drop('Price').sort_values(by='Price', ascending=False)

sns.heatmap(
    price_corr, 
    annot=True, 
    fmt='.3f', 
    cmap='RdYlGn', 
    linewidths=1.2, 
    cbar_kws={"label": "Correlation with Price (r)"}, 
    vmin=-1, vmax=1
)
plt.title("Feature Correlation with House Price", fontsize=13, fontweight='bold', pad=15)
plt.ylabel("Features", fontsize=11, fontweight='semibold')
plt.tight_layout()
plt.savefig('price_drivers_heatmap.png', dpi=200)
plt.close()
print("[OK] Saved 'price_drivers_heatmap.png'")

# 3. Pivot Table Heatmap: Location Score vs Bedrooms on Mean Price (₹ Millions)
plt.figure(figsize=(9, 7))
pivot_price = df.pivot_table(
    index='Location_score', 
    columns='Bedrooms', 
    values='Price', 
    aggfunc='mean'
) / 1e6

sns.heatmap(
    pivot_price, 
    annot=True, 
    fmt='.2f', 
    cmap='YlOrRd', 
    linewidths=1, 
    cbar_kws={"label": "Mean Price in Millions (₹)"}
)
plt.title("Average House Price by Location Score & Bedrooms (₹ Millions)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Bedrooms (BHK)", fontsize=11, fontweight='semibold')
plt.ylabel("Location Rating Score (1-10)", fontsize=11, fontweight='semibold')
plt.tight_layout()
plt.savefig('location_bedroom_price_heatmap.png', dpi=200)
plt.close()
print("[OK] Saved 'location_bedroom_price_heatmap.png'")

# 4. Hierarchical Clustermap
g = sns.clustermap(
    corr_matrix, 
    cmap='vlag', 
    linewidths=0.5, 
    figsize=(11, 11), 
    cbar_kws={"label": "Correlation (r)"}, 
    vmin=-1, vmax=1
)
g.fig.suptitle("Hierarchical Clustermap of Feature Relationships", fontsize=15, fontweight='bold', y=1.02)
plt.savefig('hierarchical_clustermap.png', dpi=200)
plt.close()
print("[OK] Saved 'hierarchical_clustermap.png'")

print("\nAll 4 heatmaps generated successfully in Python!")
