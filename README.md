# Evaluation-of-AI-techniques-for-predicting-etreme-precipitation-in-RJ-based-on-MTS-data
Repository providing  **source code, experimental configurations, final dataset and aggregated results** used in experiments to evaluate ML models for predicting extreme precipitation events in the municipality of Rio de Janeiro, based on multivariate time series data.


This repository is related to **Chapters 5 and 6** of the following MSc dissertation:

> **Macedo, Rafael Garcia.**
> *Avaliação de Técnicas de Inteligência Artificial para Previsão de Eventos Extremos de Precipitação no Rio de Janeiro a Partir de Dados de Séries Temporais Multivariadas*  
> (Evaluation of Artificial Intelligence Techniques for Extreme Rainfall Forecasting in Rio de Janeiro from Multivariate Time Series Data).  
> MSc Dissertation – Fluminense Federal University (UFF), 2026.

The main objective of this repository is to **promote transparency, reproducibility, and scientific reuse**, allowing other researchers to understand, reproduce, and extend the conducted experiments.

---

## 🎯 Repository Scope

This repository exclusively covers the stages related to:

### 📌 Chapter 5 — Data Extraction, Pre-processing, and Integration
- Extraction of public observational data
- Treatment of inconsistencies, outliers, and missing values
- Integration of multiple meteorological data sources
- Feature engineering and feature selection
- Dataset preparation for machine learning models

### 📌 Chapter 6 — Prediction Experiments
- Temperature forecasting (model validation experiments)
- Rainfall forecasting across multiple intensity ranges
- Comparison of several ML and DL architectures
- Predictive, computational, and energy efficiency evaluation
- Anomaly detection using Auto Encoder models

<!-- ❗ **This repository does not include the dissertation text nor the complete raw datasets**, due to legal and size constraints. However, all scripts required to reproduce the full pipeline from the original public data sources are provided.
-->
---

## 🌧️ Problem Statement

This work investigates **very short-term forecasting (nowcasting)** of extreme rainfall events in the city of Rio de Janeiro using **Multivariate Time Series (MTS)** derived from heterogeneous observational sources.

The task is modeled as a **univariate regression problem**, where:
- The input is a multivariate time series
- The output is rainfall accumulation predicted over a 12-hour forecasting horizon

Rainfall intensity classes follow the [**Alerta Rio**](https://www.sistema-alerta-rio.com.br/previsao-do-tempo/termosmet/) alert system:

| Intensity | Rainfall Rate |
|---------|---------------|
| Light | ≤ 5 mm/h |
| Moderate | 5 – 25 mm/h |
| Heavy | 25 – 50 mm/h |
| Very Heavy | > 50 mm/h |

---

## 📊 Data Sources

The experiments rely solely on **public observational datasets**, organized as multivariate time series:

- **Meteorological Stations** (temperature, pressure, humidity, wind, precipitation, solar radiation)
  - [INMET](https://portal.inmet.gov.br/)
  - [GEORIO](https://georio.prefeitura.rio/)  

- **Meteoceanographic Buoys** (Sea surface temperature)
  - [SIMCosta](https://simcosta.furg.br/)

- **Atmospheric Soundings** (Radiosonde data - Meteorological variables form multiple standard pressure levels)
  - [IGRA/Galeão Air Force Base](https://www.decea.mil.br/?i=midia-e-informacao&p=pg_noticia&materia=conheca-o-servico-de-radiossondagem-realizado-pelo-dtcea-galeao)

All extraction, cleaning, and integration procedures are implemented and documented in this repository.

---

## 🧠 Evaluated Models

The following Machine Learning and Deep Learning models were evaluated:

### Baseline Models
- Artificial Neural Networks (ANN)
- XGBoost Regressor 

### Sequential Models
- 1D Convolutional Neural Network (1D-CNN)
- Long Short-Term Memory (LSTM)
- CNN-LSTM

### Decomposition-ANN
- [DLinear](10.1609/aaai.v37i9.26317.)

### Transformer-Based Models
- [PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol)

### Anomaly Detection Models
- 1D-CNN-based Auto Encoder

---

## ⚙️ Experimental Pipeline

The experimental workflow can be summarized as follows:

1. Data extraction and standardization  
2. Outlier removal and source-specific corrections  
3. Temporal and spatial data integration  
4. Multivariate missing data imputation  
5. Feature engineering and feature importance ealuation
6. Feature Selection
7. Sliding window generation (MTS format)  
8. Dataset balancing  
9. Model training and validation  
10. Predictive and energy efficiency evaluation  

<!-- 
A detailed pipeline description is available in `docs/pipeline.md`.
-->
---

## 📂 Repository Structure

```text
├── 1 - Data Extraction and Preprocessing/      
    ├── 1.1 - Meteorological Stations/           
            ├── 1.1.1 - INMET Stations 
            ├── 1.1.2 - GEORIO Stations
    ├── 1.2 - Meteorological Buoys
    ├── 1.3 - Radiosoundings
├── 2 - Data Integration, Data Imputation and Feature Engineering
├── 3 - Feature Importance Evaluation
├── 4 - Feature Selection, Tensor Generation and Data Balancing
├── 5 - Models implematation, experiments and results

(Coming soon: 
  - Result analysis procedures (metrics calculation)
  - Experiments for Anomaly Detection, with Auto Encoders)
```
NOTE: For simplicity, codes are published as Markdown (.md) files, wich can be easyly converted to Jupyter Notebooks executable files using [`jupytex`](https://jupytext.org/) lib.

```code
pip install jupytext

jupytext --to notebook my_markdown_file.md
```
