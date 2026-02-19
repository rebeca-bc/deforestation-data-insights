# 🌳 Deforestation and Its Hidden Drivers

A regression-based analysis examining which country-level socio-economic indicators are associated with annual deforestation.

Each year, millions of hectares of forest are lost worldwide. While deforestation is often linked to agriculture or development, the broader socio-economic patterns behind it are less obvious. This project merges two public datasets (World Bank indicators + Our World in Data deforestation records) across 106 countries to model and predict annual forest loss.

Two modeling approaches are used:

- **Linear model (LASSO):** identifies which variables are most relevant and estimates their direction and magnitude.
- **Non-linear model (Random Forest):** captures complex interactions and non-linear relationships to improve predictive performance.
- **OLS inference:** provides statistical significance, confidence intervals, and interpretability.

The goal is not only prediction, but interpretation: understanding which structural factors are consistently associated with deforestation across countries.

---

## Data Dictionary

| Variable                        | Source            | Description                                                |
| ------------------------------- | ----------------- | ---------------------------------------------------------- |
| `Deforestation`                 | Our World in Data | **Target** — hectares of forest lost per year              |
| `Urban_population`              | World Bank        | People living in urban areas                               |
| `Physicians per thousand`       | World Bank        | Doctors per 1,000 people (proxy for institutional quality) |
| `Density (P/Km²)`               | World Bank        | Population density                                         |
| `Total tax rate`                | World Bank        | Tax burden on businesses (% of commercial profits)         |
| `Labor force participation (%)` | World Bank        | Share of working-age population employed                   |
| `Gasoline Price`                | World Bank        | Average retail price USD/liter                             |
| `Latitude / Longitude`          | World Bank        | Geographic position of country centroid                    |
| `CPI Change (%)`                | World Bank        | Annual inflation rate                                      |
| `Infant mortality`              | World Bank        | Deaths per 1,000 live births                               |

> Full data dictionary with all 35 variables is included in the notebook.

---

## Methodology

1. Data cleaning and harmonization across sources
2. Feature engineering and skewness handling
3. Train / validation split
4. Baseline linear regression
5. LASSO feature selection
6. Random Forest modeling
7. Model comparison on validation metrics
8. OLS inference on selected predictors

---

## Results Snapshot

|                 | Linear (LASSO) | Random Forest |
| --------------- | -------------- | ------------- |
| Validation R²   | 0.475          | ~0.545        |
| Validation RMSE | 351,456 ha     | 71,279 ha     |
| Validation MAE  | 114,176 ha     | 38,604 ha     |

---

## Key Insights

- Urbanization and population density show consistent association with deforestation.
- Institutional proxies (e.g., physicians per 1,000 people) appear negatively related to forest loss.
- Geographic position (latitude) contributes non-linearly in the Random Forest model.
- Several economic variables lose significance once multicollinearity is addressed.

---

## Limitations

- Cross-sectional design limits causal interpretation.
- Country-level aggregation masks regional variation.
- Potential omitted variable bias (e.g., agricultural policy, trade dynamics).

---

## Files

| File                                                                                                             | Description                                                     |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`deforestation-regression.ipynb`](deforestation-regression.ipynb)                                               | Full Jupyter Notebook. Analysis, models, inference, conclusions |
| [`deforestation-regression.html`](deforestation-regression.html)                                                 | Static HTML export. Readable without running code               |
| [`data/world-data/world-data-2023.csv`](data/world-data/world-data-2023.csv)                                     | World Bank socio-economic indicators (via Kaggle)               |
| [`data/deforestation/annual-deforestation.csv`](data/deforestation/annual-deforestation.csv)                     | Annual deforestation by country (Our World in Data)             |
| [`data/deforestation/annual-deforestation.metadata.json`](data/deforestation/annual-deforestation.metadata.json) | Metadata for the deforestation dataset                          |

---

## Interactive Dashboard (In process - beta)

An optional Streamlit dashboard (`dashboard.py`) is included as a predictive visual explorer. It loads the raw data, retrains both models, and lets you adjust sliders for each of the 9 features to see how predicted deforestation changes in real time.

**To run it:**

```bash
pip install streamlit plotly scikit-learn pandas numpy
streamlit run dashboard.py
```

It opens in your browser at `localhost:8501`. Models train automatically on first load (~2 seconds) and stay cached while the app is running.

**Important — error range:** the predictions shown carry a large margin of error. The Random Forest validation MAE is ~38,600 ha and the linear model's is ~114,000 ha. For context, most countries in the dataset deforest between 4,000 and 76,000 ha/year — so predictions for low-deforestation countries are rough estimates, not precise figures. The dashboard is best used to explore the _direction_ of relationships (what happens when urban population increases, or physicians per thousand drops) rather than to produce exact forecasts.

Mexico Example with the data gotten from the csv and got a 52,000 aproximate prediction of hectares per year. Design of the streamlit visual: 

<img width="1440" height="900" alt="Screen Shot 2026-02-19 at 0 24 35" src="https://github.com/user-attachments/assets/44009e00-94f1-4c7d-bec1-3b83c674d1fc" />



---

## Tools

`Python` · `pandas` · `numpy` · `scikit-learn` · `statsmodels` · `matplotlib` · `seaborn` · `streamlit` · `plotly`
