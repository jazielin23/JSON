# Dataiku Model Analysis

This repository contains exported Dataiku machine learning models. Below is a comprehensive analysis of each model including model types, performance metrics, and origin information.

## Models Overview

| Model | Algorithm | Target Variable | R² | MAPE | Training Dataset |
|-------|-----------|-----------------|-----|------|------------------|
| predict-app_target | Random Forest | app_target | 0.79 | 2.7% | IncentivePointMatrix_with_arrivals_clean_for_jake |
| predict-lead_nws_wdw | Ridge Regression | target_netsales | 0.97 | 4.6% | NWS_from_morning_report_joined_w_clean_app |
| predict-netcontracts_target | OLS | netcontracts_target | 0.81 | 14.6% | Contract_model_paired_down_to_final_squished_flags |
| predict-tourcount_onsite_target | OLS | tourcount_onsite_target | 0.87 | 5.7% | WDW_Lead_Tours_pairedDownToFinal_Squished_Flags |

---

## Detailed Model Information

### 1. predict-app_target-weights

| Attribute | Value |
|-----------|-------|
| **Model Type** | Random Forest Regression |
| **Model Name** | Random forest (s172) |
| **Target Variable** | `app_target` |
| **Training Dataset** | `IncentivePointMatrix_with_arrivals_clean_for_jake` |
| **Trained Date** | 2025-07-28 |
| **Dataiku DSS Version** | 13.4.2 |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| R² | 0.7917 |
| RMSE | 7.68 |
| MAE | 6.01 |
| MAPE | 2.71% |
| Pearson Correlation | 0.9196 |
| Explained Variance Score | 0.7944 |

#### Hyperparameters

- Trees: 100
- Max Depth: 8
- Min Samples: 1

#### Features Used

- onsite_oneadult_in_hh_pcent
- onsite_deluxeorvilla_pcent
- onsite_somekids_in_hh_pcent
- arrivals_leads_los7_pcent
- arrivals_leads_los5to6_pcent
- adr
- leads_repeat_within_1year_pcent_arrivals
- max_250_inc_ppt
- points_season_2 (dummy encoded)

---

### 2. predict-lead_nws_wdw-weights

| Attribute | Value |
|-----------|-------|
| **Model Type** | Ridge (L2) Regression |
| **Model Name** | Ridge (L2) regression (don't include tours, that is already accounted for in contracts) |
| **Target Variable** | `target_netsales` |
| **Training Dataset** | `NWS_from_morning_report_joined_w_clean_app` |
| **Trained Date** | 2025-07-14 |
| **Session Name** | don't include tours, that is already accounted for in contracts |
| **Dataiku DSS Version** | 13.4.2 |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| R² | 0.9735 |
| RMSE | 147,129.49 |
| MAE | 126,581.75 |
| MAPE | 4.56% |
| Pearson Correlation | 0.9886 |
| Explained Variance Score | 0.9754 |

#### Hyperparameters

- Alpha: 0.1 (selected via grid search from [0.1, 1.0, 3.0])

#### Features Used

- lead_app_wdw
- lead_netcontracts_wdw
- tour_regime_indicator (dummy encoded)
- spring_offer (dummy encoded)
- summer_offer (dummy encoded)
- winter_offer (dummy encoded)
- fall_offer (dummy encoded)
- holiday_offer (dummy encoded)
- last_2_weeks_of_offer (dummy encoded)

---

### 3. predict-netcontracts_target-weights

| Attribute | Value |
|-----------|-------|
| **Model Type** | Ordinary Least Squares (OLS) Regression |
| **Model Name** | Ordinary Least Squares (s4) - v1 |
| **Target Variable** | `netcontracts_target` |
| **Training Dataset** | `Contract_model_paired_down_to_final_squished_flags` |
| **Trained Date** | 2025-07-29 |
| **Dataiku DSS Version** | 13.4.2 |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| R² | 0.8093 |
| RMSE | 12.26 |
| MAE | 9.54 |
| MAPE | 14.62% |
| Pearson Correlation | 0.9023 |
| Explained Variance Score | 0.8095 |

#### Dataiku Origin

- **Origin:** Exported from Analysis
- **Full Model ID:** `A-DVCFINANCEFORECAST-RcJhx5qG-NOnFQbmX-s4-pp4-m1`
- **Project Key:** `DVCFINANCEFORECAST`
- **Analysis ID:** `RcJhx5qG`

#### Features Used

- live_product_count
- min_250_inc_ppt
- max_250_inc_ppt
- percent_days_worked_cruise
- avg_laborhours_guide
- onsite_tours
- lead_pcent_nights
- PreCovid (dummy encoded)
- Offer_flag (dummy encoded)
- fiscalquarter (dummy encoded)

---

### 4. predict-tourcount_onsite_target-weights

| Attribute | Value |
|-----------|-------|
| **Model Type** | Ordinary Least Squares (OLS) Regression |
| **Model Name** | Ordinary Least Squares (s1) |
| **Target Variable** | `tourcount_onsite_target` |
| **Training Dataset** | `WDW_Lead_Tours_pairedDownToFinal_Squished_Flags` |
| **Trained Date** | 2025-07-29 |
| **Dataiku DSS Version** | 13.4.2 |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| R² | 0.8740 |
| RMSE | 47.82 |
| MAE | 39.39 |
| MAPE | 5.66% |
| Pearson Correlation | 0.9371 |
| Explained Variance Score | 0.8740 |

#### Features Used

- avg_laborhours_dvca
- Overall_Parks_TopGate
- dvcaparks_cast_perctogoal_median_tour
- pct_DVCA_Resorts
- All_DVCACast
- Overall_Resorts_TopGate
- dvcaresorts_cast_perctogoal_median_tour
- Operational_Guest_DeluxeVilla_Incentive (dummy encoded)
- Operational_Guest_Mod_Incentive (dummy encoded)
- Operational_Guest_Value_Incentive (dummy encoded)
- ERA_And_OfferFlag (dummy encoded)

---

## Technical Details

### Common Configuration

All models share the following configuration:

- **Dataiku DSS Version:** 13.4.2
- **Backend:** PY_MEMORY (Python in-memory)
- **Task Type:** PREDICTION (Regression)
- **Weighting:** No weighting
- **Calibration:** No calibration

### Python Environment

Models were trained with:

- Python: 3.6.8
- pandas: 1.1.5
- scikit-learn: 0.20.4
- scipy: 1.2.3
- lightgbm: 3.2.1
- xgboost: 1.5.2
- statsmodels: 0.12.2

---

## Dataiku Origin Information

| Model | Has Origin Metadata | Project | Analysis ID |
|-------|---------------------|---------|-------------|
| predict-app_target | No | Unknown | Unknown |
| predict-lead_nws_wdw | No | Unknown | Unknown |
| predict-netcontracts_target | Yes | DVCFINANCEFORECAST | RcJhx5qG |
| predict-tourcount_onsite_target | No | Unknown | Unknown |

Only the `predict-netcontracts_target` model contains explicit origin metadata (`sm_origin.json`) indicating it was exported from an Analysis in the **DVCFINANCEFORECAST** project. The other models do not have this file, so their exact flow/analysis location cannot be determined from the exported artifacts alone.

---

## File Structure

Each zip file contains:

```
predict-*-weights.zip
├── model.zip
│   ├── core_params.json       # Core model parameters
│   ├── perf.json              # Performance metrics
│   ├── user_meta.json         # Model metadata and labels
│   ├── train_info.json        # Training information
│   ├── dss_pipeline_meta.json # DSS pipeline metadata
│   ├── clf.pkl                # Serialized model
│   └── ... (other files)
├── sample.py                  # Sample prediction script
└── requirements.txt           # Python dependencies
```
