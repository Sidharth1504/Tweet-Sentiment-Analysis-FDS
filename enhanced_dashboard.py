"""
Enhanced Interactive Dashboard for Tweet Sentiment Stock Analysis
Features:
- Interactive data exploration with filters
- Real-time predictions using trained models
- Key insights and visualizations
- Model performance comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Tweet Sentiment Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {color: white !important;}
    .stMetric .css-1xarl3l {color: white !important;}
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_all_data():
    """Load all saved data"""
    try:
        df_clean = pd.read_csv('df_clean.csv')
        df_clean['DATE'] = pd.to_datetime(df_clean['DATE'])
        feature_importance = pd.read_csv('feature_importance.csv')
        correlation = pd.read_csv('correlation_with_target.csv', index_col=0)
        regression_results = pd.read_csv('regression_results.csv')
        classification_results = pd.read_csv('classification_results.csv')
        pca_info = pd.read_csv('pca_info.csv')

        with open('feature_info.json', 'r') as f:
            feature_info = json.load(f)

        return {
            'df': df_clean,
            'feature_importance': feature_importance,
            'correlation': correlation,
            'regression_results': regression_results,
            'classification_results': classification_results,
            'pca_info': pca_info,
            'feature_info': feature_info
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure all CSV files are in the same directory as this dashboard.")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        rf_reg = joblib.load('best_rf_regressor.pkl')
        rf_class = joblib.load('best_rf_classifier.pkl')
        scaler = joblib.load('scaler.pkl')
        return {'regressor': rf_reg, 'classifier': rf_class, 'scaler': scaler}
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load everything
data = load_all_data()
models = load_models()

if data is None or models is None:
    st.stop()

df = data['df']
best_features = data['feature_info']['best_features']

# Main title
st.title("📈 Tweet Sentiment's Impact on Stock Returns")
st.markdown("### Interactive Analysis Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("🎛️ Dashboard Controls")

# Page selector
page = st.sidebar.selectbox(
    "📄 Select Page",
    ["🏠 Overview", "📊 Data Exploration", "🤖 Predictions", "📈 Model Performance", "💡 Key Insights"]
)

# Universal filters
st.sidebar.markdown("### 🔍 Data Filters")

# Stock filter
available_stocks = sorted(df['STOCK'].dropna().unique())
selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    options=available_stocks,
    default=available_stocks[:5] if len(available_stocks) > 5 else available_stocks
)

# Date range
min_date = df['DATE'].min().date()
max_date = df['DATE'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Sentiment filter
sentiment_options = ['Positive', 'Negative', 'Neutral']
selected_sentiments = st.sidebar.multiselect(
    "LSTM Sentiment",
    options=sentiment_options,
    default=sentiment_options
)

# Apply filters
filtered_df = df.copy()
if selected_stocks:
    filtered_df = filtered_df[filtered_df['STOCK'].isin(selected_stocks)]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['DATE'] >= pd.to_datetime(date_range[0])) & 
        (filtered_df['DATE'] <= pd.to_datetime(date_range[1]))
    ]
if selected_sentiments:
    filtered_df = filtered_df[filtered_df['LSTM_SENTIMENT'].isin(selected_sentiments)]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

# ===== PAGE 1: OVERVIEW =====
if page == "🏠 Overview":
    st.header("📋 Dataset Overview")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Tweets", f"{len(filtered_df):,}", 
                 delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None)

    with col2:
        st.metric("Unique Stocks", f"{filtered_df['STOCK'].nunique()}")

    with col3:
        avg_return = filtered_df['1_DAY_RETURN'].mean() * 100
        st.metric("Avg 1-Day Return", f"{avg_return:.3f}%")

    with col4:
        avg_volatility = filtered_df['VOLATILITY_10D'].mean()
        st.metric("Avg Volatility", f"{avg_volatility:.2f}")

    with col5:
        pos_sentiment_pct = (filtered_df['LSTM_SENTIMENT'] == 'Positive').sum() / len(filtered_df) * 100
        st.metric("Positive Sentiment", f"{pos_sentiment_pct:.1f}%")

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = filtered_df['LSTM_SENTIMENT'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='LSTM Sentiment Breakdown',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Return Distribution")
        fig = px.histogram(
            filtered_df,
            x='1_DAY_RETURN',
            nbins=50,
            title='1-Day Return Distribution',
            color_discrete_sequence=['#667eea']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Return")
        st.plotly_chart(fig, use_container_width=True)

    # Stock performance table
    st.subheader("🏆 Top Stocks Performance")

    stock_stats = filtered_df.groupby('STOCK').agg({
        '1_DAY_RETURN': ['mean', 'std', 'count'],
        'VOLATILITY_10D': 'mean',
        'LSTM_POLARITY': 'mean'
    }).round(4)
    stock_stats.columns = ['Avg_Return', 'Std_Return', 'Tweet_Count', 'Avg_Volatility', 'Avg_Sentiment']
    stock_stats = stock_stats.sort_values('Avg_Return', ascending=False).head(10)

    st.dataframe(stock_stats.style.background_gradient(cmap='RdYlGn', subset=['Avg_Return']), 
                use_container_width=True)

    # Time series
    st.subheader("📅 Tweet Activity Over Time")

    daily_tweets = filtered_df.groupby(filtered_df['DATE'].dt.date).size().reset_index()
    daily_tweets.columns = ['Date', 'Count']

    fig = px.line(
        daily_tweets,
        x='Date',
        y='Count',
        title='Daily Tweet Volume',
        markers=True
    )
    fig.update_traces(line_color='#667eea')
    st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 2: DATA EXPLORATION =====
elif page == "📊 Data Exploration":
    st.header("📊 Interactive Data Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Scatter Plots", "📦 Box Plots", "🔥 Heatmaps", "📉 Correlations"])

    with tab1:
        st.subheader("Scatter Plot Analysis")

        col1, col2 = st.columns(2)

        numeric_cols = ['LAST_PRICE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN',
                       '7_DAY_RETURN', 'PX_VOLUME_LOG', 'VOLATILITY_10D', 'VOLATILITY_30D',
                       'LSTM_POLARITY', 'TEXTBLOB_POLARITY']

        with col1:
            x_var = st.selectbox("X-axis Variable", numeric_cols, index=8)
        with col2:
            y_var = st.selectbox("Y-axis Variable", numeric_cols, index=1)

        color_var = st.selectbox("Color By", ['LSTM_SENTIMENT', 'STOCK', 'VOLATILITY_CATEGORY'])

        sample_size = st.slider("Sample Size", 100, min(5000, len(filtered_df)), 2000)
        sample_df = filtered_df.sample(n=min(sample_size, len(filtered_df)))

        fig = px.scatter(
            sample_df,
            x=x_var,
            y=y_var,
            color=color_var,
            title=f'{x_var} vs {y_var}',
            trendline="ols",
            hover_data=['STOCK', 'DATE']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show correlation
        corr_val = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
        st.info(f"**Correlation between {x_var} and {y_var}: {corr_val:.4f}**")

    with tab2:
        st.subheader("Box Plot Comparisons")

        cat_var = st.selectbox("Categorical Variable", ['LSTM_SENTIMENT', 'TEXTBLOB_SENTIMENT', 'VOLATILITY_CATEGORY', 'STOCK'])
        num_var = st.selectbox("Numeric Variable", numeric_cols, index=1, key='box_num')

        if cat_var == 'STOCK':
            # Limit to top stocks
            top_stocks = filtered_df['STOCK'].value_counts().head(10).index
            plot_df = filtered_df[filtered_df['STOCK'].isin(top_stocks)]
        else:
            plot_df = filtered_df

        fig = px.box(
            plot_df,
            x=cat_var,
            y=num_var,
            color=cat_var,
            title=f'{num_var} by {cat_var}'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap")

        corr_matrix = filtered_df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top Correlations with 1-Day Return:**")
        return_corr = corr_matrix['1_DAY_RETURN'].sort_values(ascending=False)
        st.write(return_corr)

    with tab4:
        st.subheader("Feature Importance & Correlations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Feature Importance (Random Forest)**")
            fig = px.bar(
                data['feature_importance'],
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance Ranking',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Correlation with Target**")
            corr_df = pd.DataFrame({
                'Feature': data['correlation'].index,
                'Correlation': data['correlation'].values
            }).sort_values('Correlation', ascending=False)

            fig = px.bar(
                corr_df,
                x='Correlation',
                y='Feature',
                orientation='h',
                title='Feature Correlations with Returns',
                color='Correlation',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 3: PREDICTIONS =====
elif page == "🤖 Predictions":
    st.header("🤖 Stock Return Predictions")
    st.markdown("Use the trained Random Forest models to predict stock returns")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Input Features")
        st.markdown("Enter values for prediction:")

        user_inputs = {}
        for feature in best_features:
            if feature in filtered_df.columns:
                mean_val = filtered_df[feature].mean()
                std_val = filtered_df[feature].std()
                min_val = filtered_df[feature].min()
                max_val = filtered_df[feature].max()

                user_inputs[feature] = st.number_input(
                    f"{feature}",
                    value=float(mean_val),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    step=float(std_val/10),
                    format="%.6f"
                )

        predict_button = st.button("🎯 Make Prediction", type="primary")

    with col2:
        st.subheader("📊 Prediction Results")

        if predict_button:
            # Prepare input
            input_df = pd.DataFrame([user_inputs])

            # Regression prediction
            reg_pred = models['regressor'].predict(input_df)[0]

            # Classification prediction
            class_pred = models['classifier'].predict(input_df)[0]
            class_proba = models['classifier'].predict_proba(input_df)[0]

            # Display results
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric(
                    "Predicted Return",
                    f"{reg_pred*100:.3f}%",
                    delta="Regression Model"
                )

            with col_b:
                st.metric(
                    "Return Category",
                    class_pred,
                    delta="Classification Model"
                )

            with col_c:
                confidence = max(class_proba) * 100
                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )

            # Probability distribution
            st.markdown("**Class Probabilities:**")
            prob_df = pd.DataFrame({
                'Class': ['Negative', 'Neutral', 'Positive'],
                'Probability': class_proba
            })

            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                title='Prediction Confidence by Class',
                color='Probability',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature contribution
            st.markdown("**Your Input vs Dataset Average:**")
            comparison_df = pd.DataFrame({
                'Feature': best_features,
                'Your Value': [user_inputs[f] for f in best_features],
                'Dataset Avg': [filtered_df[f].mean() for f in best_features],
                'Difference': [user_inputs[f] - filtered_df[f].mean() for f in best_features]
            })
            st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Difference']),
                        use_container_width=True)

        else:
            st.info("👈 Enter feature values and click 'Make Prediction'")

            # Show example predictions
            st.markdown("**Example Recent Predictions:**")
            sample_recent = filtered_df.head(5)[best_features]
            recent_preds = models['regressor'].predict(sample_recent)

            sample_recent['Predicted_Return'] = recent_preds * 100
            sample_recent['Actual_Return'] = filtered_df.head(5)['1_DAY_RETURN'].values * 100
            sample_recent['Error'] = sample_recent['Predicted_Return'] - sample_recent['Actual_Return']

            st.dataframe(sample_recent.round(4), use_container_width=True)

# ===== PAGE 4: MODEL PERFORMANCE =====
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Regression Models", "🎯 Classification Models", "📉 Detailed Metrics"])

    with tab1:
        st.subheader("Regression Model Comparison")

        reg_results = data['regression_results']

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                reg_results,
                x='Model',
                y='R2_Score',
                title='R² Score Comparison',
                color='R2_Score',
                color_continuous_scale='Viridis',
                text='R2_Score'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                reg_results,
                x='Model',
                y='RMSE',
                title='RMSE Comparison',
                color='RMSE',
                color_continuous_scale='Reds',
                text='RMSE'
            )
            fig.update_traces(texttemplate='%{text:.6f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        # Best model highlight
        best_model = reg_results.loc[reg_results['R2_Score'].idxmax()]
        st.success(f"🏆 **Best Regression Model:** {best_model['Model']} with R² = {best_model['R2_Score']:.6f}")

        st.dataframe(reg_results.style.highlight_max(subset=['R2_Score'], color='lightgreen')
                    .highlight_min(subset=['RMSE'], color='lightgreen'),
                    use_container_width=True)

    with tab2:
        st.subheader("Classification Model Comparison")

        class_results = data['classification_results']

        fig = px.bar(
            class_results,
            x='Model',
            y='Accuracy',
            title='Classification Accuracy Comparison',
            color='Accuracy',
            color_continuous_scale='Blues',
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.add_hline(y=0.333, line_dash="dash", line_color="red", 
                     annotation_text="Random Baseline (33.3%)")
        st.plotly_chart(fig, use_container_width=True)

        # Best model
        best_class_model = class_results.loc[class_results['Accuracy'].idxmax()]
        st.success(f"🏆 **Best Classification Model:** {best_class_model['Model']} with Accuracy = {best_class_model['Accuracy']:.6f}")

        st.dataframe(class_results.style.highlight_max(subset=['Accuracy'], color='lightblue'),
                    use_container_width=True)

    with tab3:
        st.subheader("PCA Analysis")

        pca_info = data['pca_info']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Scree Plot', 'Cumulative Variance Explained')
        )

        fig.add_trace(
            go.Scatter(x=pca_info['Component'], y=pca_info['Explained_Variance'],
                      mode='lines+markers', name='Variance'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=pca_info['Component'], y=pca_info['Cumulative_Variance'],
                      mode='lines+markers', name='Cumulative', line=dict(color='red')),
            row=1, col=2
        )

        fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=2,
                     annotation_text="80% Threshold")

        fig.update_xaxes(title_text="Principal Component", row=1, col=1)
        fig.update_xaxes(title_text="Principal Component", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        n_components_80 = (pca_info['Cumulative_Variance'] >= 0.8).idxmax() + 1
        st.info(f"**{n_components_80} components** needed to explain 80% of variance")

# ===== PAGE 5: KEY INSIGHTS =====
elif page == "💡 Key Insights":
    st.header("💡 Key Insights & Findings")

    # Insight boxes
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Main Finding</h3>
        <p><strong>Tweet sentiment shows a measurable but modest impact on stock returns.</strong></p>
        <p>While statistically significant, sentiment explains only 8-68% of return variance depending on the model used. 
        This suggests that sentiment should be used as a <strong>supplementary indicator</strong> rather than a standalone prediction tool.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>📊 Model Performance</h4>
            <ul>
                <li><strong>Best Regression:</strong> Random Forest (R² = 0.68)</li>
                <li><strong>Best Classification:</strong> Random Forest (89% accuracy)</li>
                <li>Tree-based models outperform linear models</li>
                <li>Non-linear relationships dominate</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4>🔍 Feature Importance</h4>
            <ul>
                <li><strong>Top Predictor:</strong> Volatility measures</li>
                <li>Sentiment has weak individual correlation (0.06-0.12)</li>
                <li>Volume and price interact with sentiment</li>
                <li>Temporal effects show momentum patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.subheader("📈 Key Visualizations")

    tab1, tab2, tab3 = st.tabs(["Sentiment vs Returns", "Top Features", "Model Comparison"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.box(
                filtered_df,
                x='LSTM_SENTIMENT',
                y='1_DAY_RETURN',
                color='LSTM_SENTIMENT',
                title='Returns Distribution by Sentiment',
                color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Correlation scatter
            sample = filtered_df.sample(min(3000, len(filtered_df)))
            fig = px.scatter(
                sample,
                x='LSTM_POLARITY',
                y='1_DAY_RETURN',
                color='VOLATILITY_CATEGORY',
                title='Sentiment Polarity vs Returns (by Volatility)',
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            data['feature_importance'].head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='Importance',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Combine results
        all_models = pd.concat([
            data['regression_results'][['Model', 'R2_Score']].assign(Type='Regression'),
            data['classification_results'][['Model', 'Accuracy']].assign(Type='Classification').rename(columns={'Accuracy': 'R2_Score'})
        ])

        fig = px.bar(
            all_models,
            x='Model',
            y='R2_Score',
            color='Type',
            title='All Models Performance Comparison',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Conclusions
    st.subheader("📝 Conclusions & Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ What Works:**
        - Ensemble methods (Random Forest) perform best
        - Combining sentiment with volatility improves predictions
        - Short-term returns (1-day) are more predictable
        - Momentum effects exist across time horizons
        """)

    with col2:
        st.markdown("""
        **⚠️ Limitations:**
        - Sentiment alone has weak predictive power
        - Dataset concentrated on few stocks
        - Missing MENTION column limits analysis
        - LSTM sentiment lacks neutral category
        """)

    st.markdown("""
    <div class="insight-box">
        <h4>🚀 Future Improvements</h4>
        <ol>
            <li><strong>Data Enhancement:</strong> Include tweet metadata (retweets, user influence)</li>
            <li><strong>Advanced Models:</strong> Try BERT-based financial sentiment models</li>
            <li><strong>Real-time Analysis:</strong> Implement live tweet streaming and prediction</li>
            <li><strong>Risk Management:</strong> Develop volatility-adjusted trading strategies</li>
            <li><strong>Broader Coverage:</strong> Expand to more stocks and longer time periods</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Final summary metrics
    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Size", f"{len(df):,} tweets")
    with col2:
        best_r2 = data['regression_results']['R2_Score'].max()
        st.metric("Best R²", f"{best_r2:.4f}")
    with col3:
        best_acc = data['classification_results']['Accuracy'].max()
        st.metric("Best Accuracy", f"{best_acc:.4f}")
    with col4:
        avg_return = df['1_DAY_RETURN'].mean() * 100
        st.metric("Avg Return", f"{avg_return:.3f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Tweet Sentiment Stock Analysis Dashboard</strong></p>
    <p>Data Science Project | October 2025</p>
    <p>Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
