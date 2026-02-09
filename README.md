# 🌍 Total Population Dashboard (1960–2024)

Bu proje, **1960–2024** yılları arasında ülkelere göre toplam nüfus verisini
keşfetmek, karşılaştırmak ve görselleştirmek için geliştirilmiştir.

Uygulama; ülke bazlı analizler, karşılaştırmalar, anomali tespiti ve
interaktif dünya haritaları sunar.

## 🚀 Özellikler
- Ülke bazlı nüfus trendi
- Yıllık büyüme oranları ve anomaliler
- Ülke karşılaştırma (normalize & log ölçek)
- Keşfet sayfası (filtreleme, sıralama, CSV indirme)
- Dünya haritası (nüfus / yüzde değişim / mutlak değişim)
- Türkçe arayüz

## 📊 Veri Kaynağı
Veri, Dünya Bankası’ndan alınmıştır:  
https://data.worldbank.org/indicator/SP.POP.TOTL

## 🛠️ Kullanılan Teknolojiler
- Python
- Streamlit
- Pandas
- Matplotlib
- Plotly

## ▶️ Lokal Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
