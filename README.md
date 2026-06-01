# Credit Risk EDA — Give Me Some Credit

## Project Overview
Exploratory data analysis on 150,000 real loan applicants to identify 
who is likely to default. Built as a multi-tool portfolio project 
using Python, SQL, and Power BI.

## Research Question
Do different analytical tools produce different insights on the same 
credit risk data?

## Dataset
- Source: Kaggle — Give Me Some Credit
- 150,000 rows, 11 features
- Target variable: SeriousDlqin2yrs (1 = defaulted, 0 = did not)

## Key Findings
- 93% of borrowers did not default — class imbalance problem identified
- Late payments (90+ days) is the strongest predictor of default
- Age, debt ratio and income alone are weak predictors
- Defaults peak in the 40-60 age group

## Tools Used
- Python (pandas, matplotlib) — EDA and data cleaning
- SQL — in progress
- Power BI — in progress

## Charts
![Default vs Non Default](default_vs_nondefault.png)
![Age vs Default](age_vs_default.png)
![Debt Ratio vs Default](debtratio_vs_default.png)
![Income vs Default](income_vs_default.png)
![Late Payments vs Default](latepayment_vs_default.png)
