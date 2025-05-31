import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Temizlenmiş veriyi yükle
df = pd.read_excel('processed_data.xlsx')

# 2. Özellikler ve hedef değişkeni ayır
X = df[['HW1', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Midterm', 'Project', 'Final']]
y = df['Category']

# 3. Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Veriyi standardize et
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. k-NN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# 6. Tahminler yap
y_pred = knn.predict(X_test)

# 7. Model performansını değerlendir
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

print("\nKarışıklık Matrisi:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 8. Görselleştirme
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Başarısız', 'Başarılı',],
            yticklabels=['Başarısız', 'Başarılı',])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('k-NN Sınıflandırma Sonuçları')
plt.show()

# 9. Optimal k değerini bulma
error_rate = []
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,20), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Hata Oranına Göre k Değeri')
plt.xlabel('k Değeri')
plt.ylabel('Hata Oranı')
plt.show()

# 10. En iyi k değeri ile final model
optimal_k = error_rate.index(min(error_rate)) + 1
print(f"\nOptimal k değeri: {optimal_k}")

final_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
final_knn.fit(X_train, y_train)
final_pred = final_knn.predict(X_test)

print("\nOptimal Model Sınıflandırma Raporu:")
print(classification_report(y_test, final_pred))


