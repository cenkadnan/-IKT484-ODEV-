import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

# CSV dosyasını yükle
df = pd.read_csv("emlak_otomobil_ulkeler_50_2010_2019.csv")

# Ülke bazında yıllık verileri özetleyelim (ortalama)
df_mean = df.groupby("Ülke").mean().reset_index()

# İlk 25 ülke otomobil ülkesi (label=1), kalan emlak ülkesi (label=0)
ulkeler = df_mean["Ülke"].tolist()
labels = [1 if i < 25 else 0 for i in range(len(ulkeler))]


df_mean["Gerçek_Sınıf"] = labels

# Özellikler (modelde kullanılacak sütunlar)
features = ["Konut Fiyat Endeksi", "İnşaat GSYH Payı (%)", "Kentsel Nüfus Oranı (%)",
            "Kişi Başına Araç Sayısı", "Otomobil Üretimi"]

X = df_mean[features]
y = df_mean["Gerçek_Sınıf"]

# Etiketleri sayısal forma dönüştür (zaten sayısal ama uyum için)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# LDA modelini eğit
lda = LinearDiscriminantAnalysis()
lda.fit(X, y_encoded)

# Modelin tahminini kendisi üzerinde deneyelim
y_pred_encoded = lda.predict(X)
y_pred = le.inverse_transform(y_pred_encoded)

df_mean["Tahmin"] = y_pred

# Sonuçları göster
print(df_mean[["Ülke", "Gerçek_Sınıf", "Tahmin"]])

# LDA bileşen katsayılarını göster (özelliklerin ayırt edicilik katkısı)
print("\nLDA bileşenleri katsayıları:")
for feature, coef in zip(features, lda.scalings_[:, 0]):
    print(f"{feature}: {coef:.4f}")


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_encoded, y_pred_encoded)
print(f"Doğruluk: {accuracy:.2f}")
