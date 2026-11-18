# 🦺 Workplace Injury Predictor – Web App

A simple web app for my INT234 Predictive Analytics project. It lets anyone explore 1 lakh+ OSHA severe injury reports, check the amputation risk of a job situation, and forecast monthly injuries.

## Pages
| Page | What it does |
|------|--------------|
| 🏠 Home | Key numbers and main findings |
| 📊 Explore the Data | Interactive charts with year and industry filters, plus the chi-square test |
| 🎯 Amputation Risk Checker | Choose a situation and get the risk %, what all 5 models predict, and safety tips |
| 📈 Monthly Injury Forecast | Linear Regression forecast of next month's injuries for any industry |
| 🏆 Model Results | Accuracy, precision, recall, F1 and confusion matrix for every model |
| 🗺️ State Groups | K-Means clusters of U.S. states, with the elbow method |

## Run on your laptop
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser at http://localhost:8501. The first load takes about 15 seconds while the models train.

## Dataset
[OSHA Severe Injury Reports](https://www.osha.gov/severeinjury), Jan 2015 – Nov 2025. A compressed copy of the needed columns is in `data/injury_data.csv.gz` (1 MB).

## Made by
Jaskaran Singh · B.Tech CSE (Data Science) · Lovely Professional University
