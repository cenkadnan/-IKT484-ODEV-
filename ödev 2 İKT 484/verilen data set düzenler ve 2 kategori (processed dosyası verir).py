import pandas as pd


# 1. Dosyayı oku
try:
    df = pd.read_excel('IKT484-Ödev22.xlsx')
    print("Dosya başarıyla okundu. İlk 5 satır:")
    print(df.head())
except Exception as e:
    print(f"Dosya okuma hatası: {e}")
    exit()

# 2. Sütun adlarını kontrol et
print("\nOrijinal sütunlar:")
print(df.columns.tolist())

# 3. Sütunları yeniden adlandır (Excel'de yapmadıysanız)
df.columns = ['Number', 'HW1', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5',
              'Midterm', 'Project', 'Final', 'Total']

# 4. Normalizasyon yap (0-10 arası quiz notlarını 0-100'e çevir)
quiz_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
for col in quiz_cols:
    df[col] = df[col].apply(lambda x: x*10 if pd.notna(x) and x <= 10 else x)

# 5. Eksik verileri kontrol et
print("\nEksik veri sayısı:")
print(df.isnull().sum())

# 6. Eksik verileri doldur (basitçe ortalama ile)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 7. Son kontrol
print("\nİşlenmiş veri özeti:")
print(df.describe())

# 8. Sınıflandırma kategorilerini oluştur
df['Category'] = pd.cut(df['Total'],
                       bins=[0, 55, 85, 100],
                       labels=['Başarısız', 'Başarılı', 'Üstün Başarılı'])

# 9. Hazır veriyi kaydet (isteğe bağlı)
df.to_excel('processed_data.xlsx', index=False)
print("\nVeri başarıyla işlendi ve kaydedildi!")