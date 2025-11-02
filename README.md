# Tweet Sentiment's Impact on Stock Returns Analysis

## Overview
This repository contains a comprehensive data science project analyzing the relationship between Twitter sentiment and stock returns. The project includes a Jupyter notebook with data cleaning, exploratory analysis, feature engineering, and machine learning model building, along with a Streamlit dashboard for interactive visualization. The dataset, sourced from Kaggle, comprises approximately 1.4 million tweets linked to stock performance metrics.

## Project Details
- **Dataset**: [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns)
  - Shape: (1,395,450, 14)
  - Key Columns: TWEET, STOCK, DATE, LAST_PRICE, 1_DAY_RETURN, 2_DAY_RETURN, 3_DAY_RETURN, 7_DAY_RETURN, PX_VOLUME, VOLATILITY_10D, VOLATILITY_30D, LSTM_POLARITY, TEXTBLOB_POLARITY
- **Tools**: Python, Jupyter Notebook, Streamlit, pandas, sklearn, matplotlib, seaborn, plotly
- **Models**: Regression (Linear, SVM, DT, RF, AB, ANN) and Classification (Logistic, SVM, DT, RF, AB, ANN)

## Hosting
- **Google Colab**: Interactive version of the notebook is available [here](https://colab.research.google.com/drive/1NvC0_AxkrHyyuttwuo8k9PKs-aVPWQ_u?usp=sharing/). Run cells to explore the analysis live.
- **Streamlit Dashboard**: To run the interactive dashboard you can clone the repo and run streamlit run .\app.py to view the interactive dashboard
  
## Acknowledgments
- Dataset from Kaggle user "thedevastator".
