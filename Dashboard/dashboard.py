import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import KeltnerChannel 
import datetime
import warnings
import pandas_datareader.data as web # For FRED data

warnings.filterwarnings("ignore")

# --- Global Configuration ---
# File Paths
DATA_CLEANED_FOLDER = r"C:\Users\Admin\Desktop\Big Data Final\Group_3_Final_Project-main\Data"

# --- Dash App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN]) 
app.title = "Group 3 - Cross Asset Financial Analysis and Forecasting of VNIndex, Gold and Bitcoin"

# Styling
chart_container_style = {'border': '1px solid #b8cce4', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px', 'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.05)'}
main_title_container_style = {'backgroundColor': '#004085', 'padding': '25px', 'border-radius': '8px', 'margin-bottom': '30px'}
main_title_text_style = {'color': 'white', 'textAlign': 'center'}
section_header_style = {'color': '#004085', 'borderBottom': '2px solid #004085', 'paddingBottom': '10px', 'marginBottom': '20px'}
tab_label_style = {"fontWeight": "bold", "padding": "10px"}
active_tab_style = {"backgroundColor": "#e7f3fe", "borderTop": "3px solid #007bff", "fontWeight": "bold"}

# Layout for empty figures
empty_fig_layout = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No data to display or error in loading.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]}}


# === TAB 1: EDA ===
def load_eda_data(asset_name):
    file_path = f"{DATA_CLEANED_FOLDER}/{asset_name}_cleaned.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df[['Date', 'Price']].rename(columns={'Price': asset_name})
        df.set_index('Date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"EDA Data Error: File not found at {file_path} for {asset_name}")
        return pd.DataFrame()

@app.callback(
    [Output('eda-return-dist-graph', 'figure'),
     Output('eda-qq-plot-graph', 'figure'),
     Output('eda-rolling-vol-graph', 'figure'),
     Output('eda-high-vol-regimes-graph', 'figure'),
     Output('eda-rolling-mean-return-graph', 'figure'),
     Output('eda-return-dist-by-asset-graph', 'figure')],
    [Input('eda-asset-dropdown', 'value')]
)
def update_eda_tab(selected_asset):
    btc_df = load_eda_data('BTC')
    vni_df = load_eda_data('VNI')
    xau_df = load_eda_data('XAU')
    
    if btc_df.empty or vni_df.empty or xau_df.empty:
        empty_fig = go.Figure(empty_fig_layout).update_layout(title_text="Data not loaded for EDA")
        return [empty_fig]*6 
    data_eda = pd.concat([btc_df, vni_df, xau_df], axis=1) 
    if data_eda.empty or selected_asset not in data_eda.columns:
        empty_fig = go.Figure(empty_fig_layout).update_layout(title_text=f"Data for {selected_asset} not available")
        return [empty_fig]*6
    returns_eda = np.log(data_eda / data_eda.shift(1)).dropna(subset=[selected_asset]) 
    if returns_eda.empty or selected_asset not in returns_eda.columns or returns_eda[selected_asset].isnull().all():
        empty_fig = go.Figure(empty_fig_layout).update_layout(title_text=f"Returns for {selected_asset} not available or all NaN")
        return [empty_fig]*6

    fig_return_dist = px.histogram(returns_eda[selected_asset].dropna(), nbins=50, title=f'{selected_asset} Return Distribution', marginal="rug", template="plotly_white")
    fig_return_dist.update_layout(bargap=0.1, title_x=0.5)
    
    if len(returns_eda[selected_asset].dropna()) < 2: 
        fig_qq = go.Figure(empty_fig_layout).update_layout(title=f'{selected_asset} Q-Q Plot (Not enough data)')
    else:
        qq_data = stats.probplot(returns_eda[selected_asset].dropna(), dist="norm")
        fig_qq = go.Figure(layout={"template":"plotly_white"})
        fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Ordered Values'))
        fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1], mode='lines', name='Fit Line'))
        fig_qq.update_layout(title=f'{selected_asset} Q-Q Plot', title_x=0.5)

    rolling_vol = returns_eda[selected_asset].rolling(window=20).std() * np.sqrt(252) 
    fig_rolling_vol = px.line(rolling_vol.dropna(), title=f'{selected_asset} 20-Day Rolling Volatility (Annualized)', template="plotly_white")
    fig_rolling_vol.update_layout(title_x=0.5)

    if rolling_vol.dropna().empty:
        fig_high_vol = go.Figure(empty_fig_layout).update_layout(title=f'{selected_asset} High-Volatility Regimes (Not enough data)')
    else:
        vol_threshold = rolling_vol.quantile(0.90)
        high_vol_regimes = rolling_vol[rolling_vol > vol_threshold]
        fig_high_vol = go.Figure(layout={"template":"plotly_white"})
        fig_high_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='Rolling Volatility'))
        fig_high_vol.add_trace(go.Scatter(x=high_vol_regimes.index, y=high_vol_regimes, mode='markers', name='High Volatility Regime', marker=dict(color='red', size=5)))
        if pd.notna(vol_threshold):
            fig_high_vol.add_hline(y=vol_threshold, line_dash="dash", line_color="red", annotation_text="90th Percentile")
        fig_high_vol.update_layout(title=f'{selected_asset} High-Volatility Regimes (90th Percentile)', title_x=0.5)
    
    rolling_mean_return = returns_eda[selected_asset].rolling(window=20).mean() * 252 
    fig_rolling_mean = px.line(rolling_mean_return.dropna(), title=f'{selected_asset} 20-Day Rolling Mean Return (Annualized)', template="plotly_white")
    fig_rolling_mean.update_layout(title_x=0.5)

    returns_melted_all_assets = np.log(data_eda.dropna() / data_eda.dropna().shift(1)).dropna().melt(var_name='Asset', value_name='Log Return') # Ensure data_eda is droppedna before melt
    fig_return_dist_asset = px.box(returns_melted_all_assets, x='Asset', y='Log Return', title='Return Distribution by Asset', template="plotly_white")
    fig_return_dist_asset.update_layout(title_x=0.5)
    
    return fig_return_dist, fig_qq, fig_rolling_vol, fig_high_vol, fig_rolling_mean, fig_return_dist_asset

# === Dữ liệu và Hàm cho TAB 2: Forecast Model ===
def load_arima_data(asset_name):
    file_path = f"{DATA_CLEANED_FOLDER}/{asset_name}_cleaned.csv"
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
        return df.dropna()
    except FileNotFoundError: return pd.DataFrame()

@app.callback(
    [Output('arima-acf-pacf-graph', 'figure'), Output('arima-forecast-graph', 'figure')],
    [Input('arima-asset-dropdown', 'value')]
)
def update_arima_plots(selected_asset):
    df_arima = load_arima_data(selected_asset)
    if df_arima.empty or len(df_arima['Log_Return']) < 50 : 
        empty_fig = go.Figure(empty_fig_layout).update_layout(title_text=f"Not enough data for ARIMA of {selected_asset}")
        return empty_fig, empty_fig
    try: acf_values = sm.tsa.acf(df_arima['Log_Return'], nlags=40, fft=False); pacf_values = sm.tsa.pacf(df_arima['Log_Return'], nlags=40, method='ols') 
    except Exception: acf_values = np.array([np.nan]*41); pacf_values = np.array([np.nan]*41) 
    fig_acf_pacf = make_subplots(rows=2, cols=1, subplot_titles=(f'ACF - {selected_asset}', f'PACF - {selected_asset}'))
    fig_acf_pacf.add_trace(go.Bar(x=np.arange(len(acf_values)), y=acf_values, name='ACF'), row=1, col=1)
    fig_acf_pacf.add_trace(go.Bar(x=np.arange(len(pacf_values)), y=pacf_values, name='PACF'), row=2, col=1)
    fig_acf_pacf.update_layout(height=600, title_text=f"ACF and PACF for {selected_asset} Log Returns", template="plotly_white", title_x=0.5)
    train_size = int(len(df_arima) * 0.8)
    if train_size < 20: fig_forecast = go.Figure(empty_fig_layout).update_layout(title=f'{selected_asset} ARIMA Forecast (Not enough training data)'); return fig_acf_pacf, fig_forecast
    train, test = df_arima['Log_Return'][:train_size], df_arima['Log_Return'][train_size:]
    p,d,q = (1,1,1); forecast = pd.Series(np.nan, index=test.index) 
    try:
        auto_model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, trace=False, error_action='ignore', D=0, max_D=0, start_P=0, start_Q=0, max_P=3, max_Q=3)
        p, d, q = auto_model.order; model = ARIMA(train, order=(p,d,q)); model_fit = model.fit(); forecast = model_fit.forecast(steps=len(test))
    except Exception: 
        try: model = ARIMA(train, order=(1,1,1)); model_fit = model.fit(); forecast = model_fit.forecast(steps=len(test))
        except Exception: pass
    fig_forecast = go.Figure(layout={"template":"plotly_white"})
    fig_forecast.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data'))
    fig_forecast.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Actual Returns'))
    fig_forecast.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='ARIMA Forecast'))
    fig_forecast.update_layout(title=f'{selected_asset} ARIMA Log Return Forecast (Order: {p},{d},{q})', title_x=0.5)
    return fig_acf_pacf, fig_forecast

def fetch_fred_data(series_id, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Fetching {series_id} from FRED, attempt {attempt + 1}...")
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            print(f"Successfully fetched {series_id}.")
            return df
        except Exception as e:
            print(f"Error fetching {series_id} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Failed to fetch {series_id} after {max_retries} attempts.")
                return pd.DataFrame() 
    return pd.DataFrame()

def load_part6_data():
    try:
        btc_df_p6 = pd.read_csv(f"{DATA_CLEANED_FOLDER}/BTC_cleaned.csv", parse_dates=['Date']).set_index('Date')
        vni_df_p6 = pd.read_csv(f"{DATA_CLEANED_FOLDER}/VNI_cleaned.csv", parse_dates=['Date']).set_index('Date')
        xau_df_p6 = pd.read_csv(f"{DATA_CLEANED_FOLDER}/XAU_cleaned.csv", parse_dates=['Date']).set_index('Date')
        for df, name in zip([btc_df_p6, vni_df_p6, xau_df_p6], ['BTC', 'VNI', 'XAU']):
            df['Return'] = df['Price'].pct_change(); df['Volatility'] = df['Return'].rolling(window=20).std(); 
        btc_df_p6.dropna(subset=['Return', 'Volatility'], inplace=True) 
        vni_df_p6.dropna(subset=['Return', 'Volatility'], inplace=True)
        xau_df_p6.dropna(subset=['Return', 'Volatility'], inplace=True)

        start_date_fred = datetime.datetime(2015, 1, 1); end_date_fred = datetime.datetime(2025, 5, 10) 
        
        dxy_data = fetch_fred_data('DTWEXBGS', start_date_fred, end_date_fred)
        cpi_data = fetch_fred_data('CPIAUCSL', start_date_fred, end_date_fred)
        fedfunds_data = fetch_fred_data('DFF', start_date_fred, end_date_fred)

        macro_components = []
        if not dxy_data.empty: macro_components.append(dxy_data.rename(columns={'DTWEXBGS': 'DXY'}))
        else: print("Warning: DXY data from FRED is empty.")
        if not cpi_data.empty: macro_components.append(cpi_data.rename(columns={'CPIAUCSL': 'CPI'}))
        else: print("Warning: CPI data from FRED is empty.")
        if not fedfunds_data.empty: macro_components.append(fedfunds_data.rename(columns={'DFF': 'FedFunds'}))
        else: print("Warning: Federal Funds Rate data from FRED is empty.")
        
        if not macro_components:
            print("No macro data fetched successfully. Macro analysis will be limited.")
            macro_df_p6 = pd.DataFrame(index=pd.date_range(start=start_date_fred, end=end_date_fred, freq='D'))
            macro_df_p6[['DXY', 'CPI', 'FedFunds']] = np.nan # Ensure columns exist even if empty
        else:
            macro_df_p6 = pd.concat(macro_components, axis=1).resample('D').ffill()
        
        macro_df_p6.dropna(how='all', inplace=True) # Drop rows where ALL macro data is NaN after ffill

        btc_merged_p6 = btc_df_p6.join(macro_df_p6, how='left'); vni_merged_p6 = vni_df_p6.join(macro_df_p6, how='left'); xau_merged_p6 = xau_df_p6.join(macro_df_p6, how='left')
        return {'BTC': btc_merged_p6.dropna(subset=['Return']), 'VNI': vni_merged_p6.dropna(subset=['Return']), 'XAU': xau_merged_p6.dropna(subset=['Return'])}
    except Exception as e: print(f"Error loading Part 6 data: {e}"); return {'BTC': pd.DataFrame(), 'VNI': pd.DataFrame(), 'XAU': pd.DataFrame()}

@app.callback(
    [Output('sentiment-corr-graph', 'figure'), Output('sentiment-pred-graph', 'figure')],
    [Input('sentiment-asset-dropdown', 'value')]
)
def update_sentiment_plots(selected_asset):
    merged_data_p6 = load_part6_data(); df_selected_p6 = merged_data_p6.get(selected_asset)
    if df_selected_p6 is None or df_selected_p6.empty:
        empty_fig = go.Figure(empty_fig_layout).update_layout(title_text=f"Not enough data for Sentiment/Macro of {selected_asset}")
        return empty_fig, empty_fig
    corr_cols = ['Return', 'Volatility'] + [col for col in ['DXY', 'CPI', 'FedFunds'] if col in df_selected_p6.columns and not df_selected_p6[col].isnull().all()]
    if len(corr_cols) < 2 : fig_corr = go.Figure(empty_fig_layout).update_layout(title=f'{selected_asset} Correlations (Not enough data)')
    else: corr_matrix = df_selected_p6[corr_cols].corr(); fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", title=f'{selected_asset} Correlations with Macro Factors', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template="plotly_white").update_layout(title_x=0.5)
    fig_pred = go.Figure(layout={"template":"plotly_white"}); macro_features = [col for col in ['DXY', 'CPI', 'FedFunds'] if col in df_selected_p6.columns and not df_selected_p6[col].isnull().all()]
    if macro_features and 'Return' in df_selected_p6.columns and not df_selected_p6['Return'].isnull().all():
        df_pred_data = df_selected_p6[macro_features + ['Return']].dropna()
        if len(df_pred_data) > len(macro_features) + 1 : 
            X = df_pred_data[macro_features].values[:-1]; y = df_pred_data['Return'].shift(-1).dropna().values 
            if len(X) > 0 and len(y) > 0:
                X = X[:len(y)] 
                if X.shape[0] > X.shape[1]: 
                    model_lr = LinearRegression().fit(X, y); y_pred = model_lr.predict(X)
                    fig_pred.add_trace(go.Scatter(x=df_pred_data.index[-len(y):], y=y, mode='lines', name='Actual Next Day Return'))
                    fig_pred.add_trace(go.Scatter(x=df_pred_data.index[-len(y):], y=y_pred, mode='lines', name='Predicted Next Day Return (Macro)'))
                    fig_pred.update_layout(title=f'{selected_asset} Actual vs. Predicted Next Day Return (Macro Factors)', title_x=0.5)
                else: fig_pred.update_layout(title=f'Not enough samples for regression - {selected_asset}', title_x=0.5)
            else: fig_pred.update_layout(title=f'Not enough aligned X,y data - {selected_asset}', title_x=0.5)
        else: fig_pred.update_layout(title=f'Not enough valid data points for regression - {selected_asset}', title_x=0.5)
    else: fig_pred.update_layout(title=f'Missing macro features or return data for {selected_asset}', title_x=0.5)
    return fig_corr, fig_pred

# === Dữ liệu và Hàm cho TAB 3: Portfolio Creation ===
INITIAL_CAPITAL_P8 = 1_000_000_000; VNI_POINT_MULTIPLIER_P8 = 100_000  
MIN_TRADE_BTC_P8 = 0.0001; MIN_TRADE_XAU_P8 = 0.01; MIN_TRADE_VNI_CONTRACTS_P8 = 0.01 
TRAIN_START_DATE_P8 = "2015-01-01"; TRAIN_END_DATE_P8 = "2023-01-01"
BACKTEST_START_DATE_P8 = "2023-01-02"; BACKTEST_END_DATE_P8 = "2025-05-10"   
RSI_PERIOD_P8 = 14; MACD_FAST_PERIOD_P8 = 12; MACD_SLOW_PERIOD_P8 = 26; MACD_SIGNAL_PERIOD_P8 = 9
KC_WINDOW_EMA_P8 = 20; KC_WINDOW_ATR_P8 = 10; KC_MULTIPLIER_P8 = 2  
ANNUAL_RISK_FREE_RATE_P8 = 0.031 

def load_and_preprocess_cleaned_data_p8(data_path, vni_multiplier):
    asset_files_info = {'BTC': {'file': 'BTC_cleaned.csv', 'cols_original': ['Price', 'Open', 'High', 'Low', 'Vol'], 'internal_cols': ['BTC_Price_USD', 'BTC_Open_USD', 'BTC_High_USD', 'BTC_Low_USD', 'BTC_Volume']}, 'XAU': {'file': 'XAU_cleaned.csv', 'cols_original': ['Price', 'Open', 'High', 'Low', 'Vol'], 'internal_cols': ['XAU_Price_USD', 'XAU_Open_USD', 'XAU_High_USD', 'XAU_Low_USD', 'XAU_Volume']}, 'VNI': {'file': 'VNI_cleaned.csv', 'cols_original': ['Price', 'Open', 'High', 'Low', 'Vol'], 'internal_cols': ['VNI_Points', 'VNI_Open_Points', 'VNI_High_Points', 'VNI_Low_Points', 'VNI_Volume']}}
    all_data_frames = []
    for asset_prefix, info in asset_files_info.items():
        try:
            df_asset = pd.read_csv(f"{data_path}/{info['file']}", parse_dates=['Date'])
            missing_original_cols = [col for col in info['cols_original'] if col not in df_asset.columns]
            if missing_original_cols: continue
            current_asset_cols = ['Date'] + info['cols_original']; df_asset_processed = df_asset[current_asset_cols].copy()
            for col_original in info['cols_original']:
                if df_asset_processed[col_original].dtype == 'object': df_asset_processed[col_original] = df_asset_processed[col_original].astype(str).str.replace(',', '', regex=False)
                df_asset_processed[col_original] = pd.to_numeric(df_asset_processed[col_original], errors='coerce')
            rename_map = dict(zip(info['cols_original'], info['internal_cols'])); df_asset_processed.rename(columns=rename_map, inplace=True)
            df_asset_processed.set_index('Date', inplace=True); all_data_frames.append(df_asset_processed)
        except Exception: pass
    if not all_data_frames: return None
    try:
        df_usdvnd = pd.read_csv(f"{data_path}/USD_VND_exchange_rate.csv", parse_dates=['Date'])
        if 'Price' not in df_usdvnd.columns: return None
        if df_usdvnd['Price'].dtype == 'object': df_usdvnd['Price'] = df_usdvnd['Price'].astype(str).str.replace(',', '', regex=False)
        df_usdvnd['Price'] = pd.to_numeric(df_usdvnd['Price'], errors='coerce')
        df_usdvnd = df_usdvnd[['Date', 'Price']].rename(columns={'Price': 'USDVND'}).set_index('Date')
    except Exception: return None
    df_merged = df_usdvnd[['USDVND']]; 
    for df_asset_item in all_data_frames: df_merged = df_merged.join(df_asset_item, how='outer')
    df_merged = df_merged.sort_index(); df_merged['USDVND'] = df_merged['USDVND'].ffill().bfill() 
    if 'BTC_Price_USD' in df_merged.columns and 'USDVND' in df_merged.columns: df_merged['BTC_Price_VND'] = df_merged['BTC_Price_USD'] * df_merged['USDVND']
    if 'XAU_Price_USD' in df_merged.columns and 'USDVND' in df_merged.columns: df_merged['XAU_Price_VND'] = df_merged['XAU_Price_USD'] * df_merged['USDVND']
    df_merged.ffill(inplace=True); df_merged.bfill(inplace=True)
    return df_merged

def calculate_technical_indicators_p8(df):
    df_out = df.copy(); assets_ohlc_cols = {'BTC': ('BTC_Price_USD', 'BTC_High_USD', 'BTC_Low_USD'), 'XAU': ('XAU_Price_USD', 'XAU_High_USD', 'XAU_Low_USD'), 'VNI': ('VNI_Points', 'VNI_High_Points', 'VNI_Low_Points')}
    for asset_prefix, (close_col, high_col, low_col) in assets_ohlc_cols.items():
        required_cols_for_asset = [close_col, high_col, low_col]; missing_cols = [col for col in required_cols_for_asset if col not in df_out.columns or df_out[col].isnull().all()]
        if missing_cols:
            for indicator_suffix in ['_RSI', '_MACD_Line', '_MACD_Signal', '_KC_Lower', '_KC_Upper', '_KC_Mid']: df_out[f'{asset_prefix}{indicator_suffix}'] = np.nan
            continue
        close_series = pd.to_numeric(df_out[close_col], errors='coerce'); high_series = pd.to_numeric(df_out[high_col], errors='coerce'); low_series = pd.to_numeric(df_out[low_col], errors='coerce')
        if close_series.isnull().all() or high_series.isnull().all() or low_series.isnull().all(): 
            for indicator_suffix in ['_RSI', '_MACD_Line', '_MACD_Signal', '_KC_Lower', '_KC_Upper', '_KC_Mid']: df_out[f'{asset_prefix}{indicator_suffix}'] = np.nan
            continue
        rsi_indicator = RSIIndicator(close=close_series, window=RSI_PERIOD_P8, fillna=False); df_out[f'{asset_prefix}_RSI'] = rsi_indicator.rsi()
        macd_indicator = MACD(close=close_series, window_slow=MACD_SLOW_PERIOD_P8, window_fast=MACD_FAST_PERIOD_P8, window_sign=MACD_SIGNAL_PERIOD_P8, fillna=False)
        df_out[f'{asset_prefix}_MACD_Line'] = macd_indicator.macd(); df_out[f'{asset_prefix}_MACD_Signal'] = macd_indicator.macd_signal()
        kc_indicator = KeltnerChannel(high=high_series, low=low_series, close=close_series, window=KC_WINDOW_EMA_P8, window_atr=KC_WINDOW_ATR_P8, multiplier=KC_MULTIPLIER_P8, fillna=False)
        df_out[f'{asset_prefix}_KC_Upper'] = kc_indicator.keltner_channel_hband(); df_out[f'{asset_prefix}_KC_Lower'] = kc_indicator.keltner_channel_lband(); df_out[f'{asset_prefix}_KC_Mid'] = kc_indicator.keltner_channel_mband()
    indicator_cols = [col for col in df_out.columns if '_RSI' in col or '_MACD_' in col or '_KC_' in col]
    if indicator_cols: df_out[indicator_cols] = df_out[indicator_cols].fillna(method='bfill'); df_out[indicator_cols] = df_out[indicator_cols].fillna(method='ffill')
    return df_out

def generate_signals_rsi_macd_p8(df, asset_prefix):
    signals = pd.DataFrame(index=df.index); price_col_for_kelly = ''
    if asset_prefix == 'VNI': price_col_for_kelly = f'{asset_prefix}_Points' 
    elif f'{asset_prefix}_Price_VND' in df.columns: price_col_for_kelly = f'{asset_prefix}_Price_VND'
    elif f'{asset_prefix}_Price_USD' in df.columns and asset_prefix != 'VNI': price_col_for_kelly = f'{asset_prefix}_Price_USD'
    if not price_col_for_kelly or price_col_for_kelly not in df.columns or df[price_col_for_kelly].isnull().all(): signals['Signal'] = 0; return signals.assign(Price_for_Kelly=np.nan) 
    signals['Price_for_Kelly'] = df[price_col_for_kelly]; rsi_col = f'{asset_prefix}_RSI'; macd_line_col = f'{asset_prefix}_MACD_Line'; macd_signal_col = f'{asset_prefix}_MACD_Signal'
    required_indicator_cols = [rsi_col, macd_line_col, macd_signal_col]; missing_cols = [col for col in required_indicator_cols if col not in df.columns or df[col].isnull().all()]
    if missing_cols: signals['Signal'] = 0; return signals 
    buy_condition = (df[rsi_col] > 30) & (df[macd_line_col] > df[macd_signal_col]) & (df[macd_line_col].shift(1) <= df[macd_signal_col].shift(1))
    sell_condition = (df[rsi_col] < 70) & (df[macd_line_col] < df[macd_signal_col]) & (df[macd_line_col].shift(1) >= df[macd_signal_col].shift(1))
    signals['Signal'] = 0; signals.loc[buy_condition, 'Signal'] = 1; signals.loc[sell_condition, 'Signal'] = -1
    return signals 

def generate_signals_rsi_kc_p8(df, asset_prefix): 
    signals = pd.DataFrame(index=df.index); price_col_for_kelly = ''
    if asset_prefix == 'VNI': price_col_for_kelly = f'{asset_prefix}_Points'
    elif f'{asset_prefix}_Price_VND' in df.columns: price_col_for_kelly = f'{asset_prefix}_Price_VND'
    elif f'{asset_prefix}_Price_USD' in df.columns and asset_prefix != 'VNI': price_col_for_kelly = f'{asset_prefix}_Price_USD'
    if not price_col_for_kelly or price_col_for_kelly not in df.columns or df[price_col_for_kelly].isnull().all(): signals['Signal'] = 0; return signals.assign(Price_for_Kelly=np.nan)
    signals['Price_for_Kelly'] = df[price_col_for_kelly]; price_for_kc_col = f'{asset_prefix}_Price_USD' if asset_prefix != 'VNI' else f'{asset_prefix}_Points'
    if price_for_kc_col not in df.columns or df[price_for_kc_col].isnull().all(): signals['Signal'] = 0; return signals 
    rsi_col = f'{asset_prefix}_RSI'; kc_lower_col = f'{asset_prefix}_KC_Lower'; kc_upper_col = f'{asset_prefix}_KC_Upper'
    required_indicator_cols_kc = [price_for_kc_col, rsi_col, kc_lower_col, kc_upper_col]; missing_cols_kc = [col for col in required_indicator_cols_kc if col not in df.columns or df[col].isnull().all()]
    if missing_cols_kc: signals['Signal'] = 0; return signals 
    buy_condition = (df[price_for_kc_col] <= df[kc_lower_col]) & (df[rsi_col] <= 30)
    sell_condition = (df[price_for_kc_col] >= df[kc_upper_col]) & (df[rsi_col] >= 70)
    signals['Signal'] = 0; signals.loc[buy_condition, 'Signal'] = 1; signals.loc[sell_condition, 'Signal'] = -1
    return signals 

def calculate_kelly_parameters_buy_hold_sell_p8(signals_df_with_price_for_kelly):
    if signals_df_with_price_for_kelly.empty or 'Signal' not in signals_df_with_price_for_kelly.columns or 'Price_for_Kelly' not in signals_df_with_price_for_kelly.columns or signals_df_with_price_for_kelly['Price_for_Kelly'].isnull().all() or signals_df_with_price_for_kelly['Signal'].eq(0).all(): return 0, 1 
    trades = []; in_position = False; entry_price = 0
    valid_signals = signals_df_with_price_for_kelly.dropna(subset=['Price_for_Kelly', 'Signal'])
    for i in range(len(valid_signals)):
        current_signal = valid_signals['Signal'].iloc[i]; current_price = valid_signals['Price_for_Kelly'].iloc[i]
        if pd.isna(current_price): continue
        if not in_position and current_signal == 1: in_position = True; entry_price = current_price
        elif in_position and current_signal == -1:  
            if entry_price == 0: in_position = False; continue
            profit_loss_ratio = (current_price / entry_price) - 1; trades.append(profit_loss_ratio)
            in_position = False; entry_price = 0 
    if not trades: return 0, 1
    wins = [t for t in trades if t > 0]; losses = [t for t in trades if t < 0]
    if not wins or not losses:
        if not wins and not losses: return 0,1
        if not wins: p = 0
        else: p = len(wins) / len(trades)
        if not losses: b = 1000 
        else: b = 0.001 
        return p,b
    p = len(wins) / len(trades); avg_win = np.mean(wins); avg_loss = abs(np.mean(losses))
    if avg_loss == 0: b = 1000 
    else: b = avg_win / avg_loss
    if b <= 0: b = 0.001 
    return p, b

def calculate_kelly_fraction_p8(p, b):
    if b <= 0: return 0 
    f = p - ((1 - p) / b); return max(0, f) 

def get_trade_details_p8(asset_prefix, current_data_row):
    min_trade_unit = 0; price_for_trade_vnd = 0 
    if asset_prefix == 'BTC': price_for_trade_vnd = current_data_row.get('BTC_Price_VND'); min_trade_unit = MIN_TRADE_BTC_P8 
    elif asset_prefix == 'XAU': price_for_trade_vnd = current_data_row.get('XAU_Price_VND'); min_trade_unit = MIN_TRADE_XAU_P8 
    elif asset_prefix == 'VNI':
        vni_points = current_data_row.get('VNI_Points')
        if pd.notna(vni_points): price_for_trade_vnd = vni_points * VNI_POINT_MULTIPLIER_P8
        else: price_for_trade_vnd = np.nan
        min_trade_unit = MIN_TRADE_VNI_CONTRACTS_P8 
    else: raise ValueError(f"Asset prefix không xác định: {asset_prefix}")
    if pd.isna(price_for_trade_vnd) or price_for_trade_vnd <= 0: return 0, 0 
    return price_for_trade_vnd, min_trade_unit

def run_backtest_p8(full_indicator_data, initial_capital, strategy_name_key, allocation_method, fixed_kelly_params):
    if full_indicator_data is None or full_indicator_data.empty: return None
    backtest_data = full_indicator_data[(full_indicator_data.index >= BACKTEST_START_DATE_P8) & (full_indicator_data.index <= BACKTEST_END_DATE_P8)].copy()
    if backtest_data.empty: return None
    portfolio = pd.DataFrame(index=backtest_data.index)
    portfolio['Cash'] = initial_capital; portfolio['BTC_Units'] = 0.0; portfolio['XAU_Units'] = 0.0; portfolio['VNI_Units'] = 0.0 
    portfolio['BTC_Value'] = 0.0; portfolio['XAU_Value'] = 0.0; portfolio['VNI_Value'] = 0.0
    portfolio['Total_Value'] = initial_capital; portfolio['Daily_Return'] = 0.0
    assets = ['BTC', 'XAU', 'VNI']; current_positions = {asset: 0.0 for asset in assets}; asset_in_position = {asset: False for asset in assets}
    strategies_for_kelly_p8 = { 'RSI_MACD': generate_signals_rsi_macd_p8, 'RSI_KC': generate_signals_rsi_kc_p8 }
    if strategy_name_key not in strategies_for_kelly_p8: return None
    signal_generation_function = strategies_for_kelly_p8[strategy_name_key]
    asset_signals_dfs_backtest = {}
    for asset in assets:
        temp_signals = signal_generation_function(full_indicator_data, asset) 
        if 'Signal' not in temp_signals.columns: temp_signals['Signal'] = 0
        asset_signals_dfs_backtest[asset] = temp_signals.reindex(backtest_data.index).fillna({'Signal':0})
    for i in range(len(backtest_data)):
        date = backtest_data.index[i]; current_data_row = backtest_data.iloc[i]
        if i > 0:
            portfolio.loc[date, 'Cash'] = portfolio.loc[backtest_data.index[i-1], 'Cash']
            for asset in assets: current_positions[asset] = portfolio.loc[backtest_data.index[i-1], f'{asset}_Units']; asset_in_position[asset] = (current_positions[asset] > 0) 
        portfolio.loc[date, 'BTC_Units'] = current_positions['BTC']; portfolio.loc[date, 'XAU_Units'] = current_positions['XAU']; portfolio.loc[date, 'VNI_Units'] = current_positions['VNI']
        btc_price_now_vnd = current_data_row.get('BTC_Price_VND', 0); xau_price_now_vnd = current_data_row.get('XAU_Price_VND', 0); vni_points_now = current_data_row.get('VNI_Points', 0)
        portfolio.loc[date, 'BTC_Value'] = current_positions['BTC'] * (btc_price_now_vnd if pd.notna(btc_price_now_vnd) else 0)
        portfolio.loc[date, 'XAU_Value'] = current_positions['XAU'] * (xau_price_now_vnd if pd.notna(xau_price_now_vnd) else 0)
        portfolio.loc[date, 'VNI_Value'] = current_positions['VNI'] * ((vni_points_now * VNI_POINT_MULTIPLIER_P8) if pd.notna(vni_points_now) else 0)
        current_total_value = portfolio.loc[date, 'Cash'] + portfolio.loc[date, 'BTC_Value'] + portfolio.loc[date, 'XAU_Value'] + portfolio.loc[date, 'VNI_Value']
        portfolio.loc[date, 'Total_Value'] = current_total_value
        if i > 0:
            prev_total_value = portfolio.loc[backtest_data.index[i-1], 'Total_Value']
            if prev_total_value != 0 and pd.notna(prev_total_value) and pd.notna(current_total_value): portfolio.loc[date, 'Daily_Return'] = (current_total_value / prev_total_value) - 1
            else: portfolio.loc[date, 'Daily_Return'] = 0
        eligible_for_new_buy = []
        for asset_prefix_iter in assets:
            signal_df_for_asset = asset_signals_dfs_backtest.get(asset_prefix_iter)
            if signal_df_for_asset is not None and date in signal_df_for_asset.index:
                 signal_val_iter = signal_df_for_asset.loc[date, 'Signal']
                 if signal_val_iter == 1 and not asset_in_position[asset_prefix_iter]: eligible_for_new_buy.append(asset_prefix_iter)
        num_eligible_buys = len(eligible_for_new_buy) if len(eligible_for_new_buy) > 0 else 1
        for asset_prefix in assets:
            signal_df_for_asset = asset_signals_dfs_backtest.get(asset_prefix); signal_val = 0
            if signal_df_for_asset is not None and date in signal_df_for_asset.index: signal_val = signal_df_for_asset.loc[date, 'Signal']
            price_of_one_unit_vnd, min_trade_quantity = get_trade_details_p8(asset_prefix, current_data_row)
            if price_of_one_unit_vnd <= 0 or min_trade_quantity <= 0: continue
            if signal_val == 1 and not asset_in_position[asset_prefix]: 
                target_investment_vnd = 0; cash_for_this_trade = portfolio.loc[date, 'Cash'] / num_eligible_buys
                if allocation_method == 'kelly':
                    if asset_prefix in fixed_kelly_params and strategy_name_key in fixed_kelly_params[asset_prefix]:
                        p = fixed_kelly_params[asset_prefix][strategy_name_key]['p']; b = fixed_kelly_params[asset_prefix][strategy_name_key]['b']
                        kelly_f = calculate_kelly_fraction_p8(p, b); target_investment_vnd = kelly_f * cash_for_this_trade
                    else: target_investment_vnd = 0.1 * cash_for_this_trade 
                elif allocation_method == 'equal_weight': target_investment_vnd = portfolio.loc[date, 'Total_Value'] / len(assets) 
                actual_investment_vnd = min(target_investment_vnd, portfolio.loc[date, 'Cash']) 
                quantity_to_buy_float = actual_investment_vnd / price_of_one_unit_vnd if price_of_one_unit_vnd > 0 else 0
                quantity_to_buy_adjusted = np.floor(quantity_to_buy_float / min_trade_quantity) * min_trade_quantity
                cost_vnd = quantity_to_buy_adjusted * price_of_one_unit_vnd
                if cost_vnd <= portfolio.loc[date, 'Cash'] and quantity_to_buy_adjusted > 0:
                    current_positions[asset_prefix] += quantity_to_buy_adjusted; portfolio.loc[date, 'Cash'] -= cost_vnd; asset_in_position[asset_prefix] = True
            elif signal_val == -1 and asset_in_position[asset_prefix]: 
                quantity_to_sell = current_positions[asset_prefix] 
                if quantity_to_sell > 0 :
                    proceeds_vnd = quantity_to_sell * price_of_one_unit_vnd
                    current_positions[asset_prefix] = 0; portfolio.loc[date, 'Cash'] += proceeds_vnd; asset_in_position[asset_prefix] = False
            portfolio.loc[date, f'{asset_prefix}_Units'] = current_positions[asset_prefix]
        btc_val_final = current_positions['BTC'] * (current_data_row.get('BTC_Price_VND',0) if pd.notna(current_data_row.get('BTC_Price_VND')) else 0)
        xau_val_final = current_positions['XAU'] * (current_data_row.get('XAU_Price_VND',0) if pd.notna(current_data_row.get('XAU_Price_VND')) else 0)
        vni_points_final = current_data_row.get('VNI_Points',0)
        vni_val_final = current_positions['VNI'] * ((vni_points_final * VNI_POINT_MULTIPLIER_P8) if pd.notna(vni_points_final) else 0)
        portfolio.loc[date, 'BTC_Value'] = btc_val_final; portfolio.loc[date, 'XAU_Value'] = xau_val_final; portfolio.loc[date, 'VNI_Value'] = vni_val_final
        portfolio.loc[date, 'Total_Value'] = portfolio.loc[date, 'Cash'] + btc_val_final + xau_val_final + vni_val_final
        if i > 0:
            prev_total_value = portfolio.loc[backtest_data.index[i-1], 'Total_Value']; current_day_total_value = portfolio.loc[date, 'Total_Value']
            if prev_total_value != 0 and pd.notna(prev_total_value) and pd.notna(current_day_total_value): portfolio.loc[date, 'Daily_Return'] = (current_day_total_value / prev_total_value) - 1
            else: portfolio.loc[date, 'Daily_Return'] = 0
        elif i == 0 and pd.notna(portfolio.loc[date, 'Total_Value']) and portfolio.loc[date, 'Total_Value'] != initial_capital : portfolio.loc[date, 'Daily_Return'] = (portfolio.loc[date, 'Total_Value'] / initial_capital) -1
    return portfolio.dropna(subset=['Total_Value'])

def calculate_performance_metrics_p8(portfolio_df, initial_capital, annual_risk_free_rate=ANNUAL_RISK_FREE_RATE_P8): 
    if portfolio_df is None or portfolio_df.empty or 'Total_Value' not in portfolio_df.columns: return {metric: 0 for metric in ["Total Return (%)", "Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Avg Daily Portfolio Return (AR) (%)", "Avg Profit (AP) (Positive Days) (%)", "Avg Loss (AL) (Negative Days) (%)", "Volatility of Daily Returns (VL) (%)"]}
    portfolio_df['Total_Value'] = pd.to_numeric(portfolio_df['Total_Value'], errors='coerce'); portfolio_df.dropna(subset=['Total_Value'], inplace=True) 
    if portfolio_df.empty: return {metric: 0 for metric in ["Total Return (%)", "Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Avg Daily Portfolio Return (AR) (%)", "Avg Profit (AP) (Positive Days) (%)", "Avg Loss (AL) (Negative Days) (%)", "Volatility of Daily Returns (VL) (%)"]}
    total_return = (portfolio_df['Total_Value'].iloc[-1] / initial_capital - 1) * 100
    trading_days_per_year = 252; num_days = len(portfolio_df)
    if num_days < 2 : return {"Total Return (%)": total_return, "Annualized Return (%)": 0, "Annualized Volatility (%)": 0, "Sharpe Ratio": 0, "Max Drawdown (%)": 0 if total_return >=0 else total_return, "Avg Daily Portfolio Return (AR) (%)": portfolio_df['Daily_Return'].mean()*100 if num_days ==1 else 0, "Avg Profit (AP) (Positive Days) (%)": 0, "Avg Loss (AL) (Negative Days) (%)": 0, "Volatility of Daily Returns (VL) (%)": 0}
    num_years = num_days / trading_days_per_year
    annualized_return = ((portfolio_df['Total_Value'].iloc[-1] / initial_capital)**(1/num_years) - 1) * 100 if num_years > 0 else 0
    daily_returns = portfolio_df['Daily_Return'].fillna(0); annualized_volatility = daily_returns.std() * np.sqrt(trading_days_per_year) * 100
    mean_daily_return = daily_returns.mean(); std_daily_return = daily_returns.std()
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/trading_days_per_year) - 1
    excess_daily_return = mean_daily_return - daily_risk_free_rate
    sharpe_ratio = (excess_daily_return * trading_days_per_year) / (std_daily_return * np.sqrt(trading_days_per_year)) if std_daily_return != 0 else 0
    cumulative_returns = (1 + daily_returns).cumprod(); peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1; max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
    ar_portfolio = mean_daily_return * 100; positive_returns = daily_returns[daily_returns > 0]
    ap_portfolio = positive_returns.mean() * 100 if not positive_returns.empty else 0
    negative_returns = daily_returns[daily_returns < 0]; al_portfolio = negative_returns.mean() * 100 if not negative_returns.empty else 0 
    vl_portfolio = std_daily_return * 100 
    return {"Total Return (%)": total_return, "Annualized Return (%)": annualized_return, "Annualized Volatility (%)": annualized_volatility, "Sharpe Ratio": sharpe_ratio, "Max Drawdown (%)": max_drawdown, "Avg Daily Portfolio Return (AR) (%)": ar_portfolio, "Avg Profit (AP) (Positive Days) (%)": ap_portfolio, "Avg Loss (AL) (Negative Days) (%)": al_portfolio, "Volatility of Daily Returns (VL) (%)": vl_portfolio}

# --- Store results for Dash ---
def get_portfolio_data():
    raw_data_p8 = load_and_preprocess_cleaned_data_p8(DATA_CLEANED_FOLDER, VNI_POINT_MULTIPLIER_P8)
    if raw_data_p8 is None: return None, None, None
    data_with_indicators_p8 = calculate_technical_indicators_p8(raw_data_p8)
    if data_with_indicators_p8 is None: return None, None, None
    fixed_kelly_params = {}; train_data_fixed_p8 = data_with_indicators_p8[(data_with_indicators_p8.index >= TRAIN_START_DATE_P8) & (data_with_indicators_p8.index <= TRAIN_END_DATE_P8)].copy()
    strategies_for_kelly_p8 = { 'RSI_MACD': generate_signals_rsi_macd_p8, 'RSI_KC': generate_signals_rsi_kc_p8 }; assets_p8 = ['BTC', 'XAU', 'VNI']
    if not train_data_fixed_p8.empty:
        for asset in assets_p8:
            fixed_kelly_params[asset] = {}
            for strat_name, strat_func in strategies_for_kelly_p8.items():
                asset_signals_train = strat_func(train_data_fixed_p8, asset)
                if not asset_signals_train.empty and 'Signal' in asset_signals_train.columns and 'Price_for_Kelly' in asset_signals_train.columns and not asset_signals_train['Price_for_Kelly'].isnull().all():
                    p, b = calculate_kelly_parameters_buy_hold_sell_p8(asset_signals_train)
                    fixed_kelly_params[asset][strat_name] = {'p': p, 'b': b}
                else: fixed_kelly_params[asset][strat_name] = {'p': 0, 'b': 1}
    else: 
         for asset in assets_p8:
            fixed_kelly_params[asset] = {}
            for strat_name in strategies_for_kelly_p8.keys(): fixed_kelly_params[asset][strat_name] = {'p': 0, 'b': 1}
    portfolio_results = {}
    portfolio_results['RSI_MACD_Kelly'] = run_backtest_p8(data_with_indicators_p8, INITIAL_CAPITAL_P8, 'RSI_MACD', 'kelly', fixed_kelly_params)
    portfolio_results['RSI_MACD_EqualWeight'] = run_backtest_p8(data_with_indicators_p8, INITIAL_CAPITAL_P8, 'RSI_MACD', 'equal_weight', fixed_kelly_params)
    portfolio_results['RSI_KC_Kelly'] = run_backtest_p8(data_with_indicators_p8, INITIAL_CAPITAL_P8, 'RSI_KC', 'kelly', fixed_kelly_params)
    portfolio_results['RSI_KC_EqualWeight'] = run_backtest_p8(data_with_indicators_p8, INITIAL_CAPITAL_P8, 'RSI_KC', 'equal_weight', fixed_kelly_params)
    performance_summary = {}
    for name, df_pf in portfolio_results.items():
        if df_pf is not None: performance_summary[name] = calculate_performance_metrics_p8(df_pf, INITIAL_CAPITAL_P8, ANNUAL_RISK_FREE_RATE_P8)
        else: performance_summary[name] = {metric: 0 for metric in ["Total Return (%)", "Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Avg Daily Portfolio Return (AR) (%)", "Avg Profit (AP) (Positive Days) (%)", "Avg Loss (AL) (Negative Days) (%)", "Volatility of Daily Returns (VL) (%)"]}
    return fixed_kelly_params, portfolio_results, performance_summary

fixed_kelly_params_data, portfolio_results_data, performance_summary_data = get_portfolio_data()

# --- Hàm vẽ biểu đồ cho Dash ---
def plot_portfolio_value_dash(portfolio_df, strategy_title):
    fig = go.Figure(layout={"template":"plotly_white"}); 
    if portfolio_df is not None and not portfolio_df.empty and 'Total_Value' in portfolio_df.columns: fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df['Total_Value'], mode='lines', name=strategy_title))
    fig.update_layout(title=f'Portfolio Value: {strategy_title}', xaxis_title='Date', yaxis_title='Portfolio Value (VND)', title_x=0.5); return fig

def plot_combined_normalized_portfolio_values_dash(strategy_portfolio_dfs, chart_start_date, chart_end_date, initial_capital):
    fig = go.Figure(layout={"template":"plotly_white"}); colors = {'RSI_MACD_Kelly': 'royalblue', 'RSI_MACD_EqualWeight': 'forestgreen', 'RSI_KC_Kelly': 'darkorange', 'RSI_KC_EqualWeight': 'purple'}
    line_styles = {'RSI_MACD_Kelly': 'solid', 'RSI_MACD_EqualWeight': 'dash', 'RSI_KC_Kelly': 'dot', 'RSI_KC_EqualWeight': 'dashdot'}
    chart_start_dt = pd.to_datetime(chart_start_date); chart_end_dt = pd.to_datetime(chart_end_date); has_data = False
    for name, df_orig in strategy_portfolio_dfs.items():
        if df_orig is not None and not df_orig.empty and 'Total_Value' in df_orig.columns:
            df_filtered = df_orig[(df_orig.index >= chart_start_dt) & (df_orig.index <= chart_end_dt)].copy()
            if not df_filtered.empty:
                first_val = df_filtered['Total_Value'].iloc[0]
                if first_val != 0 and pd.notna(first_val): norm_vals = (df_filtered['Total_Value'] / first_val) * initial_capital
                else: norm_vals = df_filtered['Total_Value']
                fig.add_trace(go.Scatter(x=df_filtered.index, y=norm_vals, mode='lines', name=name, line=dict(color=colors.get(name), dash=line_styles.get(name)))); has_data = True
    if not has_data: return go.Figure(layout={"template":"plotly_white"}).update_layout(title="No data to display")
    fig.update_layout(title=f"Normalized Portfolio Value Comparison ({chart_start_date} to {chart_end_date})", xaxis_title='Date', yaxis_title='Normalized Portfolio Value (VND)', legend_title_text='Strategy', title_x=0.5); return fig

def plot_asset_allocation_dash(portfolio_df, strategy_title):
    fig = go.Figure(layout={"template":"plotly_white"})
    if portfolio_df is not None and not portfolio_df.empty and 'Total_Value' in portfolio_df.columns and portfolio_df['Total_Value'].abs().sum() > 0:
        alloc_df = pd.DataFrame(index=portfolio_df.index)
        alloc_df['BTC'] = (portfolio_df['BTC_Value'] / portfolio_df['Total_Value']) * 100; alloc_df['XAU'] = (portfolio_df['XAU_Value'] / portfolio_df['Total_Value']) * 100
        alloc_df['VNI'] = (portfolio_df['VNI_Value'] / portfolio_df['Total_Value']) * 100; alloc_df['Cash'] = (portfolio_df['Cash'] / portfolio_df['Total_Value']) * 100
        alloc_df = alloc_df.replace([np.inf, -np.inf], np.nan).fillna(0) 
        colors = {'BTC': '#1f77b4', 'XAU': '#ff7f0e', 'VNI': '#2ca02c', 'Cash': '#d3d3d3'}
        for asset in ['BTC', 'XAU', 'VNI', 'Cash']:
            if asset in alloc_df.columns: fig.add_trace(go.Scatter(x=alloc_df.index, y=alloc_df[asset], stackgroup='one', name=f'{asset} Allocation', line_shape='hv', fillcolor=colors.get(asset), line=dict(color=colors.get(asset))))
    fig.update_layout(title=f'Asset Allocation Over Time: {strategy_title}', xaxis_title='Date', yaxis_title='Portfolio Allocation (%)', yaxis_range=[0,100], title_x=0.5); return fig

# --- Layout của Dashboard ---
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1(
                "Group 3 - Cross Asset Financial Analysis and Forecasting of VNIndex, Gold and Bitcoin", 
                className="text-center", style=main_title_text_style 
            ), 
            width=12, 
            style=main_title_container_style 
        ), 
        className="mb-4" 
    ),
    dbc.Tabs(id="main-tabs", children=[ 
        dbc.Tab(label="EDA", tab_id="tab-eda", label_style=tab_label_style, active_label_style=active_tab_style, children=[
            dbc.Row(dbc.Col(dcc.Dropdown(id='eda-asset-dropdown', options=[{'label': 'BTC Returns', 'value': 'BTC'}, {'label': 'VNI Returns', 'value': 'VNI'}, {'label': 'XAU Returns', 'value': 'XAU'}], value='BTC', clearable=False, style={'marginBottom': '20px'}))),
            dbc.Row([dbc.Col(dcc.Graph(id='eda-return-dist-graph'), width=6, style=chart_container_style), dbc.Col(dcc.Graph(id='eda-qq-plot-graph'), width=6, style=chart_container_style)]),
            dbc.Row([dbc.Col(dcc.Graph(id='eda-rolling-vol-graph'), width=6, style=chart_container_style), dbc.Col(dcc.Graph(id='eda-high-vol-regimes-graph'), width=6, style=chart_container_style)]),
            dbc.Row([dbc.Col(dcc.Graph(id='eda-rolling-mean-return-graph'), width=6, style=chart_container_style), dbc.Col(dcc.Graph(id='eda-return-dist-by-asset-graph'), width=6, style=chart_container_style)])]),
        
        dbc.Tab(label="Forecast Model", tab_id="tab-forecast", label_style=tab_label_style, active_label_style=active_tab_style, children=[
            html.H3("ARIMA Model Analysis", className="mt-3", style=section_header_style), 
            dbc.Row(dbc.Col(dcc.Dropdown(id='arima-asset-dropdown', options=[{'label': 'BTC', 'value': 'BTC'},{'label': 'VNI', 'value': 'VNI'},{'label': 'XAU', 'value': 'XAU'}], value='BTC', clearable=False, style={'marginBottom': '20px'}))),
            dbc.Row([dbc.Col(dcc.Graph(id='arima-acf-pacf-graph'), width=12, style=chart_container_style)]), 
            dbc.Row([dbc.Col(dcc.Graph(id='arima-forecast-graph'), width=12, style=chart_container_style)]), html.Hr(style={'borderColor': '#004085', 'borderWidth': '2px'}),
            html.H3("Sentiment and Macroeconomic Factor Integration", className="mt-3", style=section_header_style), 
            dbc.Row(dbc.Col(dcc.Dropdown(id='sentiment-asset-dropdown', options=[{'label': 'BTC', 'value': 'BTC'},{'label': 'VNI', 'value': 'VNI'},{'label': 'XAU', 'value': 'XAU'}], value='BTC', clearable=False, style={'marginBottom': '20px'}))),
            dbc.Row([dbc.Col(dcc.Graph(id='sentiment-corr-graph'), width=6, style=chart_container_style), dbc.Col(dcc.Graph(id='sentiment-pred-graph'), width=6, style=chart_container_style)])]),
        
        dbc.Tab(label="Portfolio Creation", tab_id="tab-portfolio", label_style=tab_label_style, active_label_style=active_tab_style, children=[ 
            html.H3("Kelly Criterion Parameters (Training Period)", className="mt-3", style=section_header_style), 
            dbc.Row(dbc.Col(html.Div(id='kelly-params-table-div'), width=12, style=chart_container_style)), 
            html.H3("Portfolio Value Backtesting", className="mt-4", style=section_header_style), 
            dbc.Row(dbc.Col(dcc.Graph(id='combined-portfolio-value-graph'), width=12, style=chart_container_style)), 
            dbc.Row([dbc.Col(dcc.Graph(id='pnl-rsi-macd-kelly-graph'), width=6, style=chart_container_style), dbc.Col(dcc.Graph(id='pnl-rsi-macd-equal-graph'), width=6, style=chart_container_style)]),
            dbc.Row([dbc.Col(dcc.Graph(id='pnl-rsi-kc-kelly-graph'), width=6, style=chart_container_style), dbc.Col(dcc.Graph(id='pnl-rsi-kc-equal-graph'), width=6, style=chart_container_style)]),
            html.H3("Performance Metrics Summary (Backtest Period)", className="mt-4", style=section_header_style), 
            dbc.Row(dbc.Col(html.Div(id='performance-metrics-table-div'), width=12, style=chart_container_style)),
            html.H3("Asset Allocation Over Time (Backtest Period)", className="mt-4", style=section_header_style),
            dcc.Dropdown(id='allocation-strategy-dropdown', options=[{'label': 'RSI+MACD Kelly', 'value': 'RSI_MACD_Kelly'},{'label': 'RSI+MACD Equal Weight', 'value': 'RSI_MACD_EqualWeight'},{'label': 'RSI+KC Kelly', 'value': 'RSI_KC_Kelly'},{'label': 'RSI+KC Equal Weight', 'value': 'RSI_KC_EqualWeight'}], value='RSI_MACD_Kelly', clearable=False, style={'marginBottom': '10px'}),
            dbc.Row(dbc.Col(dcc.Graph(id='asset-allocation-graph'), width=12, style=chart_container_style))])
    ])
], fluid=True)

# --- Callbacks cho Tab 3 ---
@app.callback(
    [Output('kelly-params-table-div', 'children'), 
     Output('combined-portfolio-value-graph', 'figure'),
     Output('pnl-rsi-macd-kelly-graph', 'figure'), 
     Output('pnl-rsi-macd-equal-graph', 'figure'),
     Output('pnl-rsi-kc-kelly-graph', 'figure'), 
     Output('pnl-rsi-kc-equal-graph', 'figure'),
     Output('performance-metrics-table-div', 'children')],
    [Input('main-tabs', 'active_tab')] 
)
def update_portfolio_tab(active_tab_id): 
    if active_tab_id != 'tab-portfolio': 
        raise dash.exceptions.PreventUpdate

    empty_fig_layout_cb = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "Error loading data or no data available.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]}}
    empty_fig_cb = go.Figure(empty_fig_layout_cb)

    if fixed_kelly_params_data is None or portfolio_results_data is None or performance_summary_data is None:
        return html.P("Error loading portfolio data. Please check data loading functions and file paths."), empty_fig_cb, empty_fig_cb, empty_fig_cb, empty_fig_cb, empty_fig_cb, html.P("Error loading performance data.")
    
    kelly_df_data = []
    if fixed_kelly_params_data: 
        for asset, strategies in fixed_kelly_params_data.items():
            for strategy, params in strategies.items(): kelly_df_data.append({'Asset': asset, 'Strategy': strategy, 'Winning Rate (p)': f"{params.get('p',0)*100:.2f}%", 'Decimal Odds (b)': f"{params.get('b',0):.2f}"})
    kelly_table = dash_table.DataTable(data=kelly_df_data, columns=[{"name": i, "id": i} for i in (['Asset', 'Strategy', 'Winning Rate (p)', 'Decimal Odds (b)'] if kelly_df_data else [])], style_table={'overflowX': 'auto', 'border': '1px solid #ddd'}, style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Arial', 'border': '1px solid #ddd'}, style_header={'backgroundColor': '#e7f3fe', 'fontWeight': 'bold', 'borderBottom': '2px solid #007bff'}, style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}])
    
    fig_combined_pnl = plot_combined_normalized_portfolio_values_dash(portfolio_results_data, BACKTEST_START_DATE_P8, BACKTEST_END_DATE_P8, INITIAL_CAPITAL_P8)
    
    fig_pnl_rm_k = plot_portfolio_value_dash(portfolio_results_data.get('RSI_MACD_Kelly'), 'RSI+MACD Kelly')
    fig_pnl_rm_e = plot_portfolio_value_dash(portfolio_results_data.get('RSI_MACD_EqualWeight'), 'RSI+MACD Equal Weight')
    fig_pnl_rk_k = plot_portfolio_value_dash(portfolio_results_data.get('RSI_KC_Kelly'), 'RSI+KC Kelly')
    fig_pnl_rk_e = plot_portfolio_value_dash(portfolio_results_data.get('RSI_KC_EqualWeight'), 'RSI+KC Equal Weight')
    
    perf_df = pd.DataFrame(performance_summary_data).T.reset_index().rename(columns={'index':'Strategy'})
    
    def format_value(value, col_name):
        if pd.isna(value): return "N/A"
        if '%' in col_name: return f"{value:.2f}%"
        if col_name == 'Sharpe Ratio': return f"{value:.2f}"
        return f"{value:.4f}"

    for col in perf_df.columns:
        if col != 'Strategy':
            perf_df[col] = perf_df[col].apply(lambda x: format_value(x, col))
            
    perf_table = dash_table.DataTable(data=perf_df.to_dict('records'), columns=[{"name": i, "id": i} for i in perf_df.columns], style_table={'overflowX': 'auto', 'minWidth': '100%', 'border': '1px solid #ddd'}, style_cell={'textAlign': 'right', 'padding': '8px', 'fontFamily': 'Arial', 'minWidth': '90px', 'width': '130px', 'maxWidth': '160px', 'border': '1px solid #ddd'}, style_header={'backgroundColor': '#e7f3fe', 'fontWeight': 'bold', 'borderBottom': '2px solid #007bff'}, style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}], sort_action="native", sort_mode="multi")
    
    return kelly_table, fig_combined_pnl, fig_pnl_rm_k, fig_pnl_rm_e, fig_pnl_rk_k, fig_pnl_rk_e, perf_table

@app.callback(Output('asset-allocation-graph', 'figure'), [Input('allocation-strategy-dropdown', 'value')])
def update_allocation_chart(selected_strategy_key):
    if portfolio_results_data is None: return go.Figure()
    df_to_plot = portfolio_results_data.get(selected_strategy_key)
    return plot_asset_allocation_dash(df_to_plot, selected_strategy_key)

if __name__ == '__main__':
    app.run(debug=True)
