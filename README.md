# Group_3_Final_Project: Cross-Asset Financial Analysis and Forecasting of VNIndex, Gold and Bitcoin

## Project Structure
The repository is organized into two main sections
1.  **`Data`**: Contains all data-related files.
    * `Raw Data`: The original, unprocessed data files used in the project.
    * `Cleaned Data`: Includes 3 final cleaned datasets: BTC_cleaned.csv, XAU_cleaned.csv, VNI_cleaned.csv
    * `Data with feature engineering`: includes additional indicators such as EMA, RSI, Bollinger Bands,...
2.  **`Group_work`**: Jupyter Notebook file for the detailed analysis. This section is divided into 8 parts, each focusing on a different analytical perspective.
  
---

## Project Summary

**Cross-Asset Financial Analysis and Forecasting** is a comprehensive data science project that applies modern statistical and machine learning (ML) techniques to analyze and forecast the return behavior of three major asset classes:

- **Bitcoin (BTC)** – representing the cryptocurrency market  
- **Vietnam stock market Index (VNI)** – representing the Vietnamese equity market  
- **Gold (XAU)** – representing the commodities market

The project’s core objective is to evaluate how these assets behave, how predictable they are, and how forecasting models can be leveraged to make informed investment decisions and construct optimized portfolios. The workflow integrates financial theory, time series analysis, and machine learning in a multi-step process:

### 1. Data Preprocessing
We gather historical daily price data for BTC, VNI, and XAU from 2015 to 2025. After aligning their time series, we clean and standardize the dataset by fixing date formats, removing inconsistencies, handling missing values, duplicate rows. This step builds a solid foundation for all downstream analytics and modeling.

### 2. Exploratory Data Analysis (EDA)
We explore each asset’s statistical properties through summary metrics (mean, skewness, kurtosis, Sharpe ratio) and distribution plots. Events like market crashes or volatility spikes are identified to better understand asset-specific behavior. This stage provides contextual insights about risk, returns, and market dynamics.

### 3. Feature Engineering
We create predictive features from raw price data, including technical indicators (EMA, RSI, Bollinger Bands), lagged returns, and rolling statistics. These features are designed to improve the signal quality for forecasting return direction, while avoiding lookahead bias or data leakage.

### 4. Time Series Forecasting (Statistical Models)
We apply classical time series models like **ARIMA** to model and forecast asset returns. Walk-forward validation is used to ensure robust performance testing. These models serve as a statistical baseline and offer interpretable insights into trend and seasonality patterns across assets.

### 5. Trend Prediction with Machine Learning

To capture nonlinear relationships and improve directional forecasting, we trained several supervised classification models (**Random Forest**, **XGBoost**, **CatBoost**, **LGBM**, **AdaBoost**, **ExtraTrees**, **DecisionTree**) to predict the *direction* (up or down) of future asset returns.

Key steps included:

- Evaluate the models using Accuracy, Precision, Recall, F1-score, and AUC.
- Check the average feature importance of the models.
- Compare results across different assets.
- Build and evaluate an Ensemble model from the two best individual models.
- Use the trained ensemble model to calculate class probabilities.

### 6. Macroeconomic Factors Integration
We expand our analysis beyond price data by incorporating external influences:
- **Macroeconomic indicators** like USD Index, CPI, and interest rates 

These variables are tested for their impact on volatility and return direction, and assessed for added predictive value when included in the ML models.

### 7. Cross-Asset Comparison & Investment Insights
All findings are synthesized into a comparative framework. We evaluate each asset’s predictability, sensitivity to external factors, and volatility. These insights are positioned from an investor’s perspective to highlight the strengths, risks, and use cases for each asset in a diversified portfolio.

### 8. Portfolio Construction using the Kelly Criterion
Finally, we apply the **Kelly Criterion** to allocate capital across BTC, VNI, and XAU based on the probability of positive returns estimated from our classifiers. The performance of the Kelly-optimized portfolio is compared against equal-weight and volatility-adjusted benchmarks using backtesting.

---

## Contribution Breakdown

| Student Name            | Student ID  | Contribution (%) |
|-------------------------|-------------|------------------|
| Le Duy Phuong           | 11225208    | 12.5%            |
| Bui Hai Dang            | 11221196    | 12.5%            |
| Nguyen Vu Cuong         | 11221171    | 12.5%            |
| Nguyen Hong Hai         | 11222019    | 12.5%            |
| Bui Dieu Linh           | 11223320    | 12.5%            |
| Nguyen Thi Nhu Quynh    | 11225562    | 12.5%            |
| Vu Thi Le               | 11223298    | 12.5%            |
| Nguyen Van Giang        | 11221809    | 12.5%            |

--- 

## Setup
- Since each part was written by different people, please change the data link before running the file.
- To run the analysis code and the dashboard locally, ensure you have Python installed along with the necessary libraries. You can install the required packages using pip:
```bash
pip install pandas numpy matplotlib seaborn plotly pmdarima 
