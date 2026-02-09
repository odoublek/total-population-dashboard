import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import plotly.express as px
from pathlib import Path

# --------------------
# Page config
# --------------------
st.set_page_config(page_title="Toplam Nüfus Panosu", layout="wide")
st.title("🌍 Toplam Nüfus Panosu (1960–2024)")

BASE_DIR = Path(__file__).parent
file_path = BASE_DIR / "totalpopulation.xls"


# --------------------
# Load data
# --------------------
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

df = load_data(file_path)

ALL_YEARS = list(map(str, range(1960, 2025)))

# Ülke satırları (WB standardı: ISO3 kodlar genelde 3 harf)
df_countries = df[df["Country Code"].astype(str).str.len() == 3].copy()

# --------------------
# Türkçe kolon isimleri
# --------------------
TR_COLS = {
    "Country Name": "Ülke",
    "Country Code": "Ülke Kodu",
    "Start": "Başlangıç Nüfusu",
    "End": "Bitiş Nüfusu",
    "Abs Change": "Mutlak Değişim",
    "Pct Change": "Yüzde Değişim (%)",
    "CAGR": "Yıllık Bileşik Büyüme (CAGR)",
}

# --------------------
# Helpers
# --------------------
def human_format(x, pos=None):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:.0f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"

def calc_country_series(df_base, country_name, year_cols):
    row = df_base[df_base["Country Name"] == country_name]
    s = row[year_cols].T
    s.columns = ["Population"]
    s.index = s.index.astype(int)
    s["Population"] = pd.to_numeric(s["Population"], errors="coerce")
    return s

@st.cache_data
def compute_growth_table(df_base, start_year, end_year):
    year_cols = [str(start_year), str(end_year)]
    tmp = df_base[["Country Name", "Country Code"] + year_cols].copy()
    tmp.columns = ["Country Name", "Country Code", "Start", "End"]
    tmp["Start"] = pd.to_numeric(tmp["Start"], errors="coerce")
    tmp["End"] = pd.to_numeric(tmp["End"], errors="coerce")
    tmp = tmp.dropna(subset=["Start", "End"])
    tmp = tmp[tmp["Start"] > 0]

    years = end_year - start_year
    tmp["Abs Change"] = tmp["End"] - tmp["Start"]
    tmp["Pct Change"] = (tmp["Abs Change"] / tmp["Start"]) * 100
    tmp["CAGR"] = (tmp["End"] / tmp["Start"]) ** (1 / years) - 1 if years > 0 else 0
    return tmp, years

def find_anomalies(series_pop: pd.Series):
    yoy_abs = series_pop.diff()
    yoy_pct = series_pop.pct_change() * 100

    valid_pct = yoy_pct.dropna()
    valid_abs = yoy_abs.dropna()

    if valid_pct.empty or valid_abs.empty:
        return None

    max_pct_year = int(valid_pct.idxmax())
    min_pct_year = int(valid_pct.idxmin())

    return {
        "max_pct_year": max_pct_year,
        "max_pct": float(valid_pct.loc[max_pct_year]),
        "max_abs": float(valid_abs.loc[max_pct_year]),
        "min_pct_year": min_pct_year,
        "min_pct": float(valid_pct.loc[min_pct_year]),
        "min_abs": float(valid_abs.loc[min_pct_year]),
    }

def to_tr_growth_view(df_growth: pd.DataFrame) -> pd.DataFrame:
    out = df_growth.copy()
    out = out.rename(columns=TR_COLS)
    if "Yıllık Bileşik Büyüme (CAGR)" in out.columns:
        out["Yıllık Bileşik Büyüme (CAGR)"] = out["Yıllık Bileşik Büyüme (CAGR)"] * 100
    return out

# --------------------
# Sidebar navigation + global controls
# --------------------
page = st.sidebar.radio("📌 Menü", ["Ana Sayfa", "Ülke Analizi", "Karşılaştırma", "Keşfet", "Harita", "Hakkında"])

start_year, end_year = st.sidebar.slider(
    "Genel yıl aralığı",
    min_value=1960,
    max_value=2024,
    value=(1990, 2024),
)
year_cols_range = list(map(str, range(start_year, end_year + 1)))

growth_df, n_years = compute_growth_table(df_countries, start_year, end_year)

# --------------------
# PAGE: Ana Sayfa
# --------------------
if page == "Ana Sayfa":
    st.subheader("📊 Genel Özet")

    colA, colB, colC = st.columns(3)
    colA.metric("Ülke sayısı", f"{growth_df.shape[0]}")
    colB.metric("Dönem", f"{start_year}–{end_year}")
    colC.metric("Yıl sayısı", f"{n_years}")

    st.markdown("### 🚀 En yüksek yüzde artış (Top 10)")
    top_pct = growth_df.sort_values("Pct Change", ascending=False).head(10)[
        ["Country Name", "Start", "End", "Pct Change", "CAGR"]
    ]
    st.dataframe(to_tr_growth_view(top_pct), use_container_width=True)

    st.markdown("### 🐢 En düşük yüzde artış / azalanlar (Bottom 10)")
    bot_pct = growth_df.sort_values("Pct Change", ascending=True).head(10)[
        ["Country Name", "Start", "End", "Pct Change", "CAGR"]
    ]
    st.dataframe(to_tr_growth_view(bot_pct), use_container_width=True)

    st.markdown("### 🧱 En yüksek mutlak artış (Top 10)")
    top_abs = growth_df.sort_values("Abs Change", ascending=False).head(10)[
        ["Country Name", "Start", "End", "Abs Change", "Pct Change"]
    ]
    st.dataframe(to_tr_growth_view(top_abs), use_container_width=True)

# --------------------
# PAGE: Ülke Analizi
# --------------------
elif page == "Ülke Analizi":
    st.subheader("🔎 Ülke Analizi")

    countries = df_countries["Country Name"].unique()
    selected_country = st.selectbox("Ülke seç", countries)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_growth = st.checkbox("Yıllık büyüme % grafiği", value=True)
    with col2:
        use_log = st.checkbox("Log ölçek (Y)", value=False)
    with col3:
        tick_step = st.selectbox("X etiketi aralığı", [1, 2, 5, 10], index=2)
    with col4:
        show_anomaly = st.checkbox("Anomali paneli göster", value=True)

    series = calc_country_series(df_countries, selected_country, year_cols_range)

    # ---- Ülke Profili (son 10 yıl)
    last_window = 10
    last_start_year = max(start_year, end_year - last_window)
    last_cols = list(map(str, range(last_start_year, end_year + 1)))
    last_series = calc_country_series(df_countries, selected_country, last_cols)

    sp10 = last_series["Population"].iloc[0]
    ep10 = last_series["Population"].iloc[-1]
    years10 = end_year - last_start_year
    cagr10 = (ep10 / sp10) ** (1 / years10) - 1 if sp10 and sp10 > 0 and years10 > 0 else None
    pct10 = ((ep10 - sp10) / sp10) * 100 if sp10 and sp10 > 0 else None
    vol10 = (last_series["Population"].pct_change() * 100).std()

    st.markdown("### 📉 Nüfus trendi")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series["Population"])
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Nüfus")
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))

    # ---- Anomali yıllarını grafikte işaretle
    anomalies = None
    if show_anomaly:
        anomalies = find_anomalies(series["Population"])
        if anomalies is not None:
            y_max = series.loc[anomalies["max_pct_year"], "Population"]
            y_min = series.loc[anomalies["min_pct_year"], "Population"]

            ax.scatter([anomalies["max_pct_year"]], [y_max], zorder=5)
            ax.scatter([anomalies["min_pct_year"]], [y_min], zorder=5)

            ax.annotate(
                f"En yüksek: {anomalies['max_pct']:.2f}%",
                (anomalies["max_pct_year"], y_max),
                textcoords="offset points",
                xytext=(10, 10),
            )
            ax.annotate(
                f"En düşük: {anomalies['min_pct']:.2f}%",
                (anomalies["min_pct_year"], y_min),
                textcoords="offset points",
                xytext=(10, -15),
            )

    xticks = list(range(start_year, end_year + 1, tick_step))
    ax.set_xticks(xticks)
    if use_log:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    start_pop = series["Population"].iloc[0]
    end_pop = series["Population"].iloc[-1]
    abs_change = end_pop - start_pop
    pct_change = (abs_change / start_pop) * 100 if start_pop and start_pop > 0 else None
    years = end_year - start_year
    cagr = (end_pop / start_pop) ** (1 / years) - 1 if start_pop and start_pop > 0 and years > 0 else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Başlangıç", f"{int(start_pop):,}")
    m2.metric("Bitiş", f"{int(end_pop):,}")
    m3.metric("Mutlak değişim", f"{int(abs_change):,}")
    m4.metric("Toplam değişim", f"{pct_change:.2f}%")

    if cagr is not None:
        st.caption(f"Yıllık bileşik büyüme (CAGR): {cagr*100:.2f}%")

    st.markdown("### 🧾 Ülke Profili (Son 10 Yıl)")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{last_start_year} Başlangıç", f"{int(sp10):,}")
    p2.metric(f"{end_year} Bitiş", f"{int(ep10):,}")
    if cagr10 is not None:
        p3.metric("Son 10 Yıl CAGR", f"{cagr10*100:.2f}%")

    q1, q2, q3 = st.columns(3)
    if pct10 is not None:
        q1.metric("Son 10 Yıl Toplam %", f"{pct10:.2f}%")
    q2.metric("Son 10 Yıl Volatilite (Std)", f"{vol10:.2f}")
    q3.metric("Dönem", f"{last_start_year}–{end_year}")

    if show_anomaly:
        st.markdown("### 🚨 Anomali Paneli (Yıllık değişim uçları)")
        if anomalies is None:
            st.info("Anomali hesaplamak için yeterli veri yok.")
        else:
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("**📌 En yüksek yıllık büyüme**")
                st.write(f"Yıl: **{anomalies['max_pct_year']}**")
                st.write(f"Yıllık büyüme: **{anomalies['max_pct']:.2f}%**")
                st.write(f"Mutlak artış: **{int(anomalies['max_abs']):,}** kişi")
            with a2:
                st.markdown("**📌 En düşük yıllık büyüme (düşüş olabilir)**")
                st.write(f"Yıl: **{anomalies['min_pct_year']}**")
                st.write(f"Yıllık büyüme: **{anomalies['min_pct']:.2f}%**")
                st.write(f"Mutlak değişim: **{int(anomalies['min_abs']):,}** kişi")

    if show_growth:
        st.markdown("### 📈 Yıllık büyüme oranı (%)")
        g = series["Population"].pct_change() * 100
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(series.index, g)
        ax2.set_xlabel("Yıl")
        ax2.set_ylabel("Büyüme (%)")
        ax2.set_xticks(list(range(start_year, end_year + 1, tick_step)))
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# --------------------
# PAGE: Karşılaştırma
# --------------------
elif page == "Karşılaştırma":
    st.subheader("🆚 Ülke Karşılaştırma")

    countries = df_countries["Country Name"].unique()
    selected = st.multiselect("Karşılaştırılacak ülkeleri seç (2–6 önerilir)", countries)

    col1, col2, col3 = st.columns(3)
    with col1:
        normalize = st.checkbox("Normalize et (Başlangıç=100)", value=False)
    with col2:
        tick_step = st.selectbox("X etiketi aralığı", [1, 2, 5, 10], index=2, key="cmp_tick")
    with col3:
        use_log = st.checkbox("Log ölçek (Y)", value=False, key="cmp_log")

    if len(selected) < 1:
        st.info("En az 1 ülke seç.")
    else:
        st.markdown("### 📉 Karşılaştırmalı trend")
        fig, ax = plt.subplots(figsize=(12, 5))

        for c in selected:
            s = calc_country_series(df_countries, c, year_cols_range)["Population"]
            if normalize:
                s = (s / s.iloc[0]) * 100
            ax.plot(s.index, s.values, label=c)

        ax.set_xlabel("Yıl")
        ax.set_ylabel("Endeks (Başlangıç=100)" if normalize else "Nüfus")
        if not normalize:
            ax.yaxis.set_major_formatter(FuncFormatter(human_format))

        xticks = list(range(start_year, end_year + 1, tick_step))
        ax.set_xticks(xticks)

        if use_log and not normalize:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        st.pyplot(fig)

        years = end_year - start_year
        rows = []
        for c in selected:
            s = calc_country_series(df_countries, c, year_cols_range)["Population"]
            sp, ep = float(s.iloc[0]), float(s.iloc[-1])
            abs_ch = ep - sp
            pct_ch = (abs_ch / sp) * 100 if sp > 0 else None
            cagr = (ep / sp) ** (1 / years) - 1 if sp > 0 and years > 0 else None
            rows.append({
                "Ülke": c,
                "Başlangıç Nüfusu": int(sp),
                "Bitiş Nüfusu": int(ep),
                "Mutlak Değişim": int(abs_ch),
                "Yüzde Değişim (%)": round(pct_ch, 2) if pct_ch is not None else None,
                "Yıllık Bileşik Büyüme (CAGR)": round(cagr * 100, 2) if cagr is not None else None,
            })

        st.markdown("### 📋 Özet metrikler")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --------------------
# PAGE: Keşfet
# --------------------
elif page == "Keşfet":
    st.subheader("🔍 Keşfet (Filtrele, Sırala, İndir)")

    base = growth_df[[
        "Country Name", "Country Code", "Start", "End", "Abs Change", "Pct Change", "CAGR"
    ]].copy()
    base_tr = to_tr_growth_view(base)

    min_pop_end = int(pd.to_numeric(df_countries[str(end_year)], errors="coerce").dropna().min())
    max_pop_end = int(pd.to_numeric(df_countries[str(end_year)], errors="coerce").dropna().max())

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        q = st.text_input("Ülke ara", value="")
    with c2:
        sort_mode = st.selectbox("Sıralama", ["Yüzde Değişim (%)", "Mutlak Değişim"], index=0)
    with c3:
        direction = st.selectbox("Yön", ["Azalan → Artan", "Artan → Azalan"], index=1)

    pop_min, pop_max = st.slider(
        f"{end_year} nüfusu aralığı (bitiş yılı)",
        min_value=min_pop_end,
        max_value=max_pop_end,
        value=(min_pop_end, max_pop_end),
    )

    pct_min, pct_max = st.slider(
        "Yüzde değişim aralığı (%)",
        min_value=float(base_tr["Yüzde Değişim (%)"].min()),
        max_value=float(base_tr["Yüzde Değişim (%)"].max()),
        value=(float(base_tr["Yüzde Değişim (%)"].min()), float(base_tr["Yüzde Değişim (%)"].max())),
    )

    end_pop = df_countries[["Country Name", str(end_year)]].copy()
    end_pop.columns = ["Country Name", "EndYearPop"]
    end_pop["EndYearPop"] = pd.to_numeric(end_pop["EndYearPop"], errors="coerce")
    end_pop_tr = end_pop.rename(columns={"Country Name": "Ülke", "EndYearPop": f"{end_year} Nüfusu"})

    merged = base_tr.merge(end_pop_tr, on="Ülke", how="left")
    merged = merged.dropna(subset=[f"{end_year} Nüfusu"])

    if q.strip():
        merged = merged[merged["Ülke"].str.contains(q.strip(), case=False, na=False)]

    merged = merged[(merged[f"{end_year} Nüfusu"] >= pop_min) & (merged[f"{end_year} Nüfusu"] <= pop_max)]
    merged = merged[(merged["Yüzde Değişim (%)"] >= pct_min) & (merged["Yüzde Değişim (%)"] <= pct_max)]

    ascending = True if direction == "Azalan → Artan" else False
    merged = merged.sort_values(sort_mode, ascending=ascending)

    st.markdown("### 🏁 Hızlı Liste")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top 10 – Yüzde Değişim (%)**")
        st.dataframe(
            merged.sort_values("Yüzde Değişim (%)", ascending=False).head(10)[
                ["Ülke", "Başlangıç Nüfusu", "Bitiş Nüfusu", "Yüzde Değişim (%)", "Mutlak Değişim"]
            ],
            use_container_width=True
        )
    with t2:
        st.markdown("**Top 10 – Mutlak Değişim**")
        st.dataframe(
            merged.sort_values("Mutlak Değişim", ascending=False).head(10)[
                ["Ülke", "Başlangıç Nüfusu", "Bitiş Nüfusu", "Mutlak Değişim", "Yüzde Değişim (%)"]
            ],
            use_container_width=True
        )

    st.markdown("### 📋 Filtrelenmiş Sonuçlar")
    st.dataframe(
        merged[["Ülke", "Ülke Kodu", "Başlangıç Nüfusu", "Bitiş Nüfusu", f"{end_year} Nüfusu",
                "Mutlak Değişim", "Yüzde Değişim (%)", "Yıllık Bileşik Büyüme (CAGR)"]],
        use_container_width=True
    )

    csv = merged.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Sonuçları CSV indir",
        data=csv,
        file_name=f"keshfet_{start_year}_{end_year}.csv",
        mime="text/csv",
    )

# --------------------
# PAGE: Harita
# --------------------
elif page == "Harita":
    st.subheader("🗺️ Harita")

    mode = st.radio(
        "Harita metriği",
        ["Seçili yılda nüfus", "Seçili dönemde yüzde değişim", "Seçili dönemde mutlak değişim"],
        horizontal=True,
    )

    st.markdown("#### Filtreler")
    f1, f2, f3 = st.columns(3)
    with f1:
        only_negative = st.checkbox("Sadece negatif değişimler", value=False)
    with f2:
        top_n = st.selectbox("Top N (tabloda)", [10, 20, 50, 100], index=1)
    with f3:
        pass

    if mode == "Seçili yılda nüfus":
        year = st.slider("Yıl", 1960, 2024, 2024)
        tmp = df_countries[["Country Name", "Country Code", str(year)]].copy()
        tmp.columns = ["Ülke", "ISO3", "Nüfus"]
        tmp["Nüfus"] = pd.to_numeric(tmp["Nüfus"], errors="coerce")
        tmp = tmp.dropna(subset=["Nüfus"])

        fig = px.choropleth(
            tmp,
            locations="ISO3",
            color="Nüfus",
            hover_name="Ülke",
            title=f"{year} Yılı Ülkelere Göre Nüfus",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 📋 {year} Top {top_n} Nüfus")
        st.dataframe(tmp.sort_values("Nüfus", ascending=False).head(top_n), use_container_width=True)

    elif mode == "Seçili dönemde yüzde değişim":
        tmp = growth_df[["Country Name", "Country Code", "Pct Change", "Abs Change"]].copy()
        tmp.columns = ["Ülke", "ISO3", "Yüzde Değişim (%)", "Mutlak Değişim"]
        tmp["Yüzde Değişim (%)"] = pd.to_numeric(tmp["Yüzde Değişim (%)"], errors="coerce")
        tmp["Mutlak Değişim"] = pd.to_numeric(tmp["Mutlak Değişim"], errors="coerce")
        tmp = tmp.dropna(subset=["Yüzde Değişim (%)"])

        if only_negative:
            tmp = tmp[tmp["Yüzde Değişim (%)"] < 0]

        fig = px.choropleth(
            tmp,
            locations="ISO3",
            color="Yüzde Değişim (%)",
            hover_name="Ülke",
            title=f"{start_year}–{end_year} Döneminde Ülkelere Göre Nüfus Yüzde Değişimi",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 📋 Top {top_n} (Yüzde Değişim)")
        st.dataframe(tmp.sort_values("Yüzde Değişim (%)", ascending=False).head(top_n), use_container_width=True)

    else:
        tmp = growth_df[["Country Name", "Country Code", "Abs Change", "Pct Change"]].copy()
        tmp.columns = ["Ülke", "ISO3", "Mutlak Değişim", "Yüzde Değişim (%)"]
        tmp["Mutlak Değişim"] = pd.to_numeric(tmp["Mutlak Değişim"], errors="coerce")
        tmp["Yüzde Değişim (%)"] = pd.to_numeric(tmp["Yüzde Değişim (%)"], errors="coerce")
        tmp = tmp.dropna(subset=["Mutlak Değişim"])

        if only_negative:
            tmp = tmp[tmp["Mutlak Değişim"] < 0]

        fig = px.choropleth(
            tmp,
            locations="ISO3",
            color="Mutlak Değişim",
            hover_name="Ülke",
            title=f"{start_year}–{end_year} Döneminde Ülkelere Göre Nüfus Mutlak Değişimi",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 📋 Top {top_n} (Mutlak Değişim)")
        st.dataframe(tmp.sort_values("Mutlak Değişim", ascending=False).head(top_n), use_container_width=True)

# --------------------
# PAGE: Hakkında
# --------------------
elif page == "Hakkında":
    st.subheader("ℹ️ Hakkında")

    st.markdown("""
### Proje Hakkında
Bu uygulama, **1960–2024** arasında ülkelere göre **toplam nüfus** verisini keşfetmek, karşılaştırmak ve harita üzerinde incelemek için geliştirilmiştir.

Bu proje **eğlence** ve **analiz yeteneklerimi geliştirmek** amacıyla hazırlanmıştır.

### Veri Kaynağı
Veri, World Bank (Dünya Bankası) kaynağından alınmıştır:  
- https://data.worldbank.org/indicator/SP.POP.TOTL

### Hesaplama Tanımları
- **Mutlak Değişim** = Bitiş Nüfusu − Başlangıç Nüfusu  
- **Yüzde Değişim (%)** = (Mutlak Değişim / Başlangıç Nüfusu) × 100  
- **Yıllık Bileşik Büyüme (CAGR)** = (Bitiş / Başlangıç)^(1 / yıl_sayısı) − 1

### Geliştirici
**Ozgur Kaan Kaya**  
Kişisel web sitem: https://www.ozgurkaankaya.site
Github: https://github.com/odoublek"
""")
