import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the stock data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        df.rename(columns={'Unnamed: 0': 'Stock'}, inplace=True)

        if "942KIIFB32" in df['Stock'].values:
            df = df[df['Stock'] != "942KIIFB32"]

        price_cols = [col for col in df.columns if 'Minus_' in col]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.set_index('Stock', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None


# --- Helper Functions for Stationarity ---
def perform_adf_test(series):
    """Performs the ADF test and returns a formatted result dictionary."""
    series = series.dropna()
    if len(series) < 4:
        return {"result": "Cannot perform test: Not enough data points remain."}

    try:
        result = adfuller(series)
        output = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Number of Observations': result[3],
            'Critical Value (1%)': result[4]['1%'],
            'Critical Value (5%)': result[4]['5%'],
            'Critical Value (10%)': result[4]['10%']
        }
        if output['p-value'] <= 0.05:
            output['result'] = "Stationary (p-value <= 0.05)"
        else:
            output['result'] = "Not Stationary (p-value > 0.05)"
    except Exception as e:
        output = {"result": f"ADF test failed due to a calculation error ({type(e).__name__})."}

    return output


def get_stationarity_order(series):
    """Calculates the order of differencing required to make a series stationary."""
    series = series.dropna()
    if len(series) < 4:
        return "Not enough data points to calculate."

    try:
        if adfuller(series)[1] <= 0.05:
            return 0
    except Exception:
        return "Calculation error on original data. The series may have no variance."

    for i in [1, 2]:
        series_diff = series.diff(i).dropna()
        if len(series_diff) < 4:
            return f"Not enough data points after {i} differencing."
        try:
            if adfuller(series_diff)[1] <= 0.05:
                return i
        except Exception:
            return f"Calculation error after {i} differencing."

    return "Series is not stationary even after 2 differencing steps."


# --- Use the specified file path ---
file_path = r'C:\Users\Yoges\Documents\SDBI\SEM 3\Time Series\Historical_data.csv'
df = load_data(file_path)

# --- Main Application ---
st.title("📈 Stock Analysis Tool")

if df is not None:
    stock_list = df.index.tolist()

    # --- Step 1: Comparison ---
    st.markdown("---")
    st.header("Step 1: Compare Multiple Stocks")
    st.sidebar.header("⚙️ Step 1 Controls")

    if 'selection' not in st.session_state:
        st.session_state.selection = []

    selection = st.sidebar.multiselect(
        'Select up to 4 stocks to compare',
        options=stock_list,
        default=st.session_state.selection
    )

    if len(selection) > 4:
        st.sidebar.warning("Maximum of 4 stocks reached. Remove one to add another.")
        selection = st.session_state.selection
        st.experimental_rerun()

    st.session_state.selection = selection
    selected_stocks = st.session_state.selection

    time_interval = st.sidebar.slider(
        'Select time interval (days)', 7, 90, 30, 1,
        key='time_interval_step1',
        help="Select the number of recent days of historical data to display."
    )

    normalize_data = st.sidebar.checkbox(
        "Normalize Data", False,
        help="Show percentage growth instead of absolute price."
    )

    if not selected_stocks:
        st.info("👈 Select stocks from the sidebar to begin comparison.")
    else:
        cols_to_select = [f"Minus_{i}" for i in range(time_interval, 0, -1)]
        comparison_df = df.loc[selected_stocks, cols_to_select].copy()
        comparison_df = comparison_df.ffill(axis=1).bfill(axis=1)

        chart_title = f"Price Comparison for the Last {time_interval} Days"
        if normalize_data:
            comparison_df = comparison_df.apply(lambda x: (x / x.iloc[0] * 100) if x.iloc[0] != 0 else x, axis=1)
            chart_title = f"Normalized Performance for the Last {time_interval} Days"

        st.subheader(chart_title)
        plot_df = comparison_df.transpose()
        plot_df.index = [int(col.replace('Minus_', '-')) for col in plot_df.index]
        plot_df.index.name = "Days Before Present"
        st.line_chart(plot_df, use_container_width=True)

        with st.expander("View Raw Data for Step 1"):
            st.dataframe(comparison_df.style.format("{:.2f}", na_rep="Missing"))

    # --- Step 2: Stationarity ---
    st.markdown("---")
    st.header("Step 2: Stationarity Analysis")
    st.sidebar.header("⚙️ Step 2 Controls")

    selected_stock_single = st.sidebar.selectbox(
        'Select a single stock for analysis',
        options=[""] + stock_list
    )
    
    series_to_test = pd.Series(dtype=float)
    diff_order_manual = 0
    stock_series = pd.Series(dtype=float)

    if not selected_stock_single:
        st.info("👈 Select a single stock from the sidebar to perform stationarity analysis.")
    else:
        stock_series = df.loc[selected_stock_single].copy()
        stock_series = stock_series.ffill().bfill()
        # Use a numeric index for plotting and modeling
        stock_series.index = range(len(stock_series))

        st.subheader(f"Original Time Series Plot for {selected_stock_single}")
        st.line_chart(stock_series)
        st.markdown("---")

        st.sidebar.subheader("Stationarity Tests")
        diff_order_manual = st.sidebar.slider(
            "Apply manual differencing (d value)", 0, 5, 0,
            key='diff_slider_step2',
            help="Apply differencing to the series before testing. This value will be the 'd' in the ARIMA model."
        )

        col1, col2 = st.sidebar.columns(2)
        run_dw = col1.checkbox("DW Test")
        run_adf = col2.checkbox("ADF Test")

        if run_adf or run_dw:
            series_to_test = stock_series.diff(diff_order_manual).dropna()
            adf_results = {}

            if run_dw:
                st.markdown("#### Durbin-Watson (DW) Test")
                if len(series_to_test) < 2:
                    st.warning("Cannot perform DW test: Not enough data points remain.")
                else:
                    dw_value = durbin_watson(series_to_test)
                    st.metric("Durbin-Watson Statistic", f"{dw_value:.2f}")
                    interp = ""
                    if 1.5 <= dw_value <= 2.5: interp = "No significant autocorrelation (value is near 2)."
                    elif dw_value < 1.5: interp = "Indicates positive autocorrelation."
                    elif dw_value > 2.5: interp = "Indicates negative autocorrelation."
                    st.info(f"**Interpretation**: {interp}")

            if run_adf:
                st.markdown("#### Augmented Dickey-Fuller (ADF) Test")
                auto_diff_order = get_stationarity_order(stock_series)
                if isinstance(auto_diff_order, int):
                    st.success(f"**Automated Check**: Original series needs **{auto_diff_order}** differencing.")
                else:
                    st.warning(f"**Automated Check Failed**: {auto_diff_order}")
                st.write(f"**Test result on series with {diff_order_manual} differencing applied:**")
                adf_results = perform_adf_test(series_to_test)
                st.json(adf_results)

            if not series_to_test.empty:
                st.subheader(f"Plot After Applying {diff_order_manual} Differencing")
                st.line_chart(series_to_test)

            st.markdown("---")
            if run_adf and adf_results:
                if "Stationary" in adf_results.get("result", ""):
                    st.success("✅ Conclusion: Based on the ADF test, the data is now stationary.")
                elif "Not Stationary" in adf_results.get("result", ""):
                    st.warning("⚠️ Conclusion: Based on the ADF test, the data is still not stationary.")
        else:
            st.info("Select a test (DW or ADF) from the sidebar to view results.")

    # --- Step 3: ACF and PACF Analysis ---
    st.markdown("---")
    st.header("Step 3: ACF & PACF Analysis")
    st.sidebar.header("⚙️ Step 3 Controls")

    col3, col4 = st.sidebar.columns(2)
    run_acf = col3.checkbox("ACF Plot")
    run_pacf = col4.checkbox("PACF Plot")

    if not selected_stock_single:
        st.info("👈 Select a stock in Step 2 to enable ACF/PACF plotting.")
    elif not (run_acf or run_pacf):
         st.info("Select ACF or PACF from the sidebar to view the plots.")
    else:
        if series_to_test.empty:
            st.warning("Cannot generate plots. Please run a stationarity test in Step 2 first to generate the data.")
        else:
            if run_acf and run_pacf:
                col_acf, col_pacf = st.columns(2)
                with col_acf:
                    st.subheader("Autocorrelation (ACF)")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plot_acf(series_to_test, ax=ax, lags=40)
                    st.pyplot(fig)
                with col_pacf:
                    st.subheader("Partial Autocorrelation (PACF)")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plot_pacf(series_to_test, ax=ax, lags=40)
                    st.pyplot(fig)
            elif run_acf:
                st.subheader("Autocorrelation Function (ACF)")
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_acf(series_to_test, ax=ax, lags=40)
                st.pyplot(fig)
            elif run_pacf:
                st.subheader("Partial Autocorrelation Function (PACF)")
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_pacf(series_to_test, ax=ax, lags=40)
                st.pyplot(fig)
                
    # --- Step 4: ARIMA Forecasting ---
    st.markdown("---")
    st.header("Step 4: ARIMA Forecasting")
    st.sidebar.header("⚙️ Step 4 Controls")

    d_value = diff_order_manual
    p_default = 1
    q_default = 1
    
    st.sidebar.write(f"The differencing order `d` is set to **{d_value}** from Step 2.")
    st.sidebar.write(f"Developer defaults: `p={p_default}`, `q={q_default}`. You can adjust these below.")

    p_value = st.sidebar.slider("Select AR order (p)", 0, 5, p_default)
    q_value = st.sidebar.slider("Select MA order (q)", 0, 5, q_default)
    
    if not selected_stock_single:
        st.info("👈 Select a stock in Step 2 to enable forecasting.")
    else:
        st.subheader(f"ARIMA({p_value}, {d_value}, {q_value}) Forecast")
        
        try:
            model = ARIMA(stock_series, order=(p_value, d_value, q_value))
            model_fit = model.fit()
            
            # Get in-sample predictions
            predictions = model_fit.predict(start=0, end=len(stock_series)-1)
            
            # Get out-of-sample forecast
            forecast = model_fit.get_forecast(steps=10)
            forecast_mean = forecast.predicted_mean
            
            # --- UPDATE: Use Plotly for an interactive forecast chart ---
            fig = go.Figure()

            # Add traces for actual, predicted, and forecast
            fig.add_trace(go.Scatter(x=stock_series.index, y=stock_series, mode='lines', name='Actual', line=dict(color='royalblue')))
            fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Predicted', line=dict(color='orange', dash='dot')))
            fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='red')))

            fig.update_layout(
                title=f'Forecast for {selected_stock_single}',
                xaxis_title='Time Steps (Days)',
                yaxis_title='Price',
                legend_title='Legend'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Forecasted Values"):
                st.dataframe(forecast_mean)

        except Exception as e:
            st.error(f"ARIMA model failed to fit for order ({p_value}, {d_value}, {q_value}).")
            st.warning(f"Reason: {e}")