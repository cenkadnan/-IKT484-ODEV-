import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Verileri yükle
train_data = pd.read_csv('super_lig_egitim_verisi.csv')
test_data = pd.read_csv('super_lig_test_verisi.csv')

# Kategorik değişkenleri çıkar (LDA için sadece sayısal özellikler gerekli)
X_train = train_data.drop(['Takım', 'Sezon', 'Hafta', 'Etiket', 'Son_Puan'], axis=1)
y_train = train_data['Etiket']

X_test = test_data.drop(['Takım', 'Sezon', 'Hafta'], axis=1)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Etiketleri sayısallaştırma
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# LDA modelini oluştur ve eğit
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train_encoded)

# Eğitim verisi üzerinde çapraz doğrulama
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lda, X_train_scaled, y_train_encoded, cv=5)
print(f"Çapraz Doğrulama Doğrulukları: {scores}")
print(f"Ortalama Doğruluk: {scores.mean():.2f}")

# Eğitim verisi üzerinde tahmin yap
y_pred = lda.predict(X_train_scaled)

# Sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_train_encoded, y_pred, target_names=label_encoder.classes_))

# Karışıklık matrisi
cm = confusion_matrix(y_train_encoded, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('LDA Karışıklık Matrisi')
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()

# Test verisi üzerinde tahmin yap
test_predictions = lda.predict(X_test_scaled)
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Tahminleri test verisine ekle
test_data['Tahmin_Edilen_Etiket'] = test_predictions_labels

# Sonuçları görüntüle
print("\n2023-2024 Sezonu 17. Hafta Tahminleri:")
print(test_data[['Takım', 'Tahmin_Edilen_Etiket']].sort_values('Tahmin_Edilen_Etiket'))