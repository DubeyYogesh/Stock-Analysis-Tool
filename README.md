# 📈 Stock Analysis & Forecasting Tool

This project is an interactive **Streamlit application** for analyzing and forecasting stock market data using **time series techniques**. It provides an end-to-end workflow for stock data exploration, stationarity testing, ACF/PACF visualization, and ARIMA forecasting.

---

## 🔑 Features

### Stock Comparison
- Compare up to 4 stocks over a selected time interval.
- Option to normalize data for percentage growth analysis.
- Correlation heatmap of daily returns.

### Stationarity Analysis
- Augmented Dickey-Fuller (ADF) Test.
- Durbin-Watson (DW) Test.
- Automated differencing order suggestion.
- Interactive differencing with real-time plots.

### ACF & PACF Analysis
- Visualize Autocorrelation and Partial Autocorrelation plots.
- Helps in ARIMA parameter selection.

### ARIMA Forecasting
- Manual selection of ARIMA parameters (p, d, q).
- Auto-ARIMA functionality for best parameter search.
- 10-step future forecasting with interactive Plotly charts.
- Residual analysis including histograms, ACF, and Q-Q plots.

---

## 🛠 Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas, NumPy
- **Statistical Tests:** Statsmodels
- **Forecasting:** ARIMA (Statsmodels), Auto-ARIMA (pmdarima)
- **Visualization:** Matplotlib, Plotly

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-analysis-tool.git
cd stock-analysis-tool
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```
