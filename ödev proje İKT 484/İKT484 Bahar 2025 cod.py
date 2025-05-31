import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Veri yükleme
data = pd.read_csv('economic_indicators_2022_corrected.csv')

# Özellik seçimi
real_estate_features = [
    'construction_gdp', 'construction_employment', 'housing_price_index',
    'household_debt_to_gdp', 'real_estate_investment', 'urbanization_rate',
    'housing_loans', 'new_housing_starts'
]

auto_features = [
    'auto_production', 'auto_export', 'auto_gdp', 'auto_employment',
    'vehicle_density', 'electric_vehicle_share', 'auto_rnd_expenditure',
    'road_quality_index'
]

# Ölçeklendirme
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[real_estate_features + auto_features])
scaled_df = pd.DataFrame(scaled_data, columns=real_estate_features + auto_features)

# PCA uygulama (2 boyutlu görselleştirme için)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
principal_df['country'] = data['country']

# K-means kümeleme
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_df)
data['cluster'] = clusters
principal_df['cluster'] = clusters

# Küme merkezlerini analiz ederek kategorileri belirleme
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=real_estate_features + auto_features
)

# Her küme için emlak ve otomobil puanlarını hesapla
real_estate_scores = cluster_centers[real_estate_features].mean(axis=1)
auto_scores = cluster_centers[auto_features].mean(axis=1)

# Hangi kümenin hangi kategori olduğunu belirle
if real_estate_scores[0] - auto_scores[0] > real_estate_scores[1] - auto_scores[1]:
    cluster_categories = {0: 'Emlak Ülkesi', 1: 'Otomobil Ülkesi'}
else:
    cluster_categories = {1: 'Emlak Ülkesi', 0: 'Otomobil Ülkesi'}

# Ülkelere kategorileri ata
data['category'] = data['cluster'].map(cluster_categories)
principal_df['category'] = principal_df['cluster'].map(cluster_categories)

# Türkiye'nin sonucunu göster
turkey_result = data[data['country'] == 'Turkey'][['country', 'category']]
print("\nTürkiye'nin Kategorisi:")
print(turkey_result)

# 1. Görsel: PCA ile kümeleme sonuçları
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='category',
    data=principal_df,
    palette='viridis',
    s=100
)

# Ülke isimlerini ve kategorilerini ekle
for i in range(len(principal_df)):
    plt.text(
        principal_df.PC1[i],
        principal_df.PC2[i],
        f"{principal_df.country[i]} ({principal_df.category[i]})",
        fontsize=8
    )

plt.title('Ülkelerin Emlak ve Otomobil Sektörlerine Göre Sınıflandırılması (PCA)')
plt.xlabel('Emlak Sektörü Puanı')
plt.ylabel('Otomobil Sektörü Puanı')
plt.legend(title='Kategori')
plt.grid(True)

# 2. Görsel: Emlak vs Otomobil Skorları
data['real_estate_score'] = data[real_estate_features].mean(axis=1)
data['auto_score'] = data[auto_features].mean(axis=1)

plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    x='real_estate_score',
    y='auto_score',
    hue='category',
    data=data,
    palette='viridis',
    s=100
)

# Ülke isimlerini ve kategorilerini ekle
for i in range(len(data)):
    plt.text(
        data.real_estate_score[i],
        data.auto_score[i],
        f"{data.country[i]} ({data.category[i]})",
        fontsize=8
    )

plt.title('Ülkelerin Emlak ve Otomobil Sektör Puanları')
plt.xlabel('Emlak Sektörü Puanı')
plt.ylabel('Otomobil Sektörü Puanı')
plt.axvline(x=data['real_estate_score'].mean(), color='gray', linestyle='--')
plt.axhline(y=data['auto_score'].mean(), color='gray', linestyle='--')
plt.legend(title='Kategori')
plt.grid(True)

plt.show()

# Kategorilere göre ülke listeleri
print("\nEmlak Ülkeleri:")
print(data[data['category'] == 'Emlak Ülkesi']['country'].to_string(index=False))

print("\nOtomobil Ülkeleri:")
print(data[data['category'] == 'Otomobil Ülkesi']['country'].to_string(index=False))

# Sonuçları kaydet
data[['country', 'iso_code', 'category', 'real_estate_score', 'auto_score']].to_csv(
    'country_classification_results.csv',
    index=False
)