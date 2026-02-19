import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_and_train():
    world = pd.read_csv("data/world-data/world-data-2023.csv")
    defor = pd.read_csv("data/deforestation/annual-deforestation.csv")

    df = world.merge(defor, left_on="Country", right_on="Entity", how="inner")

    drop_admin = ["Abbreviation", "Official language", "Currency-Code", "Entity",
                  "Code", "Capital/Major City", "Calling Code", "Largest city", "Year"]
    df = df.drop(columns=[c for c in drop_admin if c in df.columns])

    for col in df.select_dtypes(include="object").columns:
        if col == "Country":
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$%,]", "", regex=True), errors="coerce")

    df = df.drop(columns=["Country"])

    log_cols = ["Density\n(P/Km2)", "Armed Forces size", "Co2-Emissions", "GDP",
                "Population", "Urban_population", "Maternal mortality ratio",
                "Infant mortality", "Deforestation", "Minimum wage"]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    df = df.fillna(df.median(numeric_only=True))

    drop_cols = ["Forested Area (%)", "Land Area(Km2)", "Fertility Rate", "Birth Rate",
                 "Life expectancy", "Maternal mortality ratio", "Population"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df["Deforestation"]
    X = df.drop(columns=["Deforestation"])

    final_cols = ["Urban_population", "Physicians per thousand", "Density\n(P/Km2)",
                  "Total tax rate", "Population: Labor force participation (%)",
                  "Gasoline Price", "Latitude", "Longitude", "CPI Change (%)"]
    final_cols = [c for c in final_cols if c in X.columns]
    X = X[final_cols]

    linear_f = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.2476, max_iter=10000))])
    linear_f.fit(X, y)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    return linear_f, rf, final_cols

linear_f, rf, final_cols = load_and_train()

st.title("🌳 Deforestation Predictor")
st.caption("Adjust the sliders to build a country profile and see how much annual deforestation the models predict.")

st.sidebar.header("Country Profile")

urban_pop  = st.sidebar.slider("Urban population (millions)", 0.5, 400.0, 20.0, 1.0)
physicians = st.sidebar.slider("Physicians per 1,000 people", 0.0, 8.0, 1.5, 0.1)
density    = st.sidebar.slider("Population density (P/km²)", 1, 2000, 100, 10)
tax        = st.sidebar.slider("Total tax rate (%)", 5, 80, 40, 1)
labor      = st.sidebar.slider("Labor force participation (%)", 30, 90, 60, 1)
gasoline   = st.sidebar.slider("Gasoline price (USD/L)", 0.1, 2.5, 1.0, 0.05)
lat        = st.sidebar.slider("Latitude", -55, 72, 15, 1)
lon        = st.sidebar.slider("Longitude", -180, 180, -80, 5)
inflation  = st.sidebar.slider("Inflation / CPI change (%)", -2.0, 50.0, 3.0, 0.5)

# build input row — log1p where needed (matches notebook preprocessing)
row = {
    "Urban_population":                          np.log1p(urban_pop * 1_000_000),
    "Physicians per thousand":                   physicians,
    "Density\n(P/Km2)":                          np.log1p(density),
    "Total tax rate":                            tax,
    "Population: Labor force participation (%)": labor,
    "Gasoline Price":                            gasoline,
    "Latitude":                                  lat,
    "Longitude":                                 lon,
    "CPI Change (%)":                            inflation,
}
X_new = pd.DataFrame([row])[final_cols]

lin_ha = max(0, np.expm1(linear_f.predict(X_new)[0]))
rf_ha  = max(0, np.expm1(rf.predict(X_new)[0]))

# results
st.subheader("Predicted annual deforestation")
col1, col2 = st.columns(2)
col1.metric("Linear (LASSO)", f"{lin_ha:,.0f} ha/year")
col2.metric("Random Forest",  f"{rf_ha:,.0f} ha/year")

avg = (lin_ha + rf_ha) / 2
if avg > 500_000:
    st.error("⚠️ HIGH RISK — profile resembles major deforestation countries")
elif avg > 50_000:
    st.warning("🟡 MODERATE RISK — notable forest pressure expected")
else:
    st.success("🟢 LOW RISK — within typical range for this profile")

st.divider()

# show which features are pushing prediction up or down (linear coefficients)
st.subheader("What's driving the prediction?")
st.caption("Based on the linear model coefficients — positive means associated with more deforestation.")

coefs = linear_f.named_steps["lasso"].coef_
labels = [
    "Urban population", "Physicians /1K", "Density",
    "Total tax rate", "Labor force", "Gasoline price",
    "Latitude", "Longitude", "Inflation"
]
df_coef = pd.DataFrame({"feature": labels[:len(coefs)], "coefficient": coefs})
df_coef = df_coef.sort_values("coefficient")

fig = px.bar(
    df_coef, x="coefficient", y="feature", orientation="h",
    color="coefficient", color_continuous_scale=["#2ecc71", "#e74c3c"],
    labels={"coefficient": "Effect on deforestation", "feature": ""},
)
fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
st.plotly_chart(fig, use_container_width=True)

st.caption("Run with: `streamlit run dashboard.py` from the project folder")
