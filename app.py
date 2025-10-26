"""
Enhanced Interactive Dashboard for Tweet Sentiment Stock Analysis
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =======================
# PAGE CONFIGURATION
# =======================
st.set_page_config(
    page_title="Tweet Sentiment Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# CUSTOM CSS STYLING
# =======================
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }

    /* Metrics styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }

    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    /* Insight boxes */
    .insight-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }

    /* Success box */
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }

    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

# =======================
# DATA LOADING FUNCTIONS
# =======================
@st.cache_data
def load_data():
    """Load all CSV data files"""
    try:
        data = {}

        # Check if files exist
        required_files = [
            'df_clean.csv',
            'feature_importance.csv',
            'correlation_with_target.csv',
            'regression_results.csv',
            'classification_results.csv',
            'pca_info.csv'
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
            st.info("Please run all modification cells in your Jupyter notebook first.")
            return None

        # Load dataframes
        data['df'] = pd.read_csv('df_clean.csv')
        data['df']['DATE'] = pd.to_datetime(data['df']['DATE'])
        data['feature_importance'] = pd.read_csv('feature_importance.csv')
        data['correlation'] = pd.read_csv('correlation_with_target.csv', index_col=0)
        data['regression_results'] = pd.read_csv('regression_results.csv')
        data['classification_results'] = pd.read_csv('classification_results.csv')
        data['pca_info'] = pd.read_csv('pca_info.csv')

        # Load feature info
        with open('feature_info.json', 'r') as f:
            data['feature_info'] = json.load(f)

        return data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models = {}

        required_models = [
            'best_rf_regressor.pkl',
            'best_rf_classifier.pkl',
            'scaler.pkl'
        ]

        missing_models = [f for f in required_models if not os.path.exists(f)]
        if missing_models:
            st.error(f"Missing model files: {', '.join(missing_models)}")
            return None

        models['regressor'] = joblib.load('best_rf_regressor.pkl')
        models['classifier'] = joblib.load('best_rf_classifier.pkl')
        models['scaler'] = joblib.load('scaler.pkl')

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# =======================
# LOAD DATA AND MODELS
# =======================
with st.spinner('Loading data and models...'):
    data = load_data()
    models = load_models()

if data is None or models is None:
    st.stop()

df = data['df']
best_features = data['feature_info']['best_features']

# =======================
# SIDEBAR NAVIGATION
# =======================
st.sidebar.image("https://img.icons8.com/color/96/000000/stock-market.png", width=80)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Overview", "📊 Data Explorer", "🤖 Predictions", "📈 Model Performance", "💡 Key Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

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

try:
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
except:
    date_range = (min_date, max_date)

# Sentiment filter
selected_sentiments = st.sidebar.multiselect(
    "LSTM Sentiment",
    options=['Positive', 'Negative', 'Neutral'],
    default=['Positive', 'Negative', 'Neutral']
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

st.sidebar.info(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")

# =======================
# PAGE 1: OVERVIEW
# =======================
if page == "🏠 Overview":
    st.title("📈 Tweet Sentiment Stock Analysis Dashboard")
    st.markdown("### Comprehensive Analysis of Tweet Sentiment's Impact on Stock Returns")
    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Tweets",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )

    with col2:
        st.metric("Unique Stocks", f"{filtered_df['STOCK'].nunique()}")

    with col3:
        avg_return = filtered_df['1_DAY_RETURN'].mean() * 100
        st.metric("Avg 1-Day Return", f"{avg_return:.3f}%")

    with col4:
        avg_vol = filtered_df['VOLATILITY_10D'].mean()
        st.metric("Avg Volatility", f"{avg_vol:.2f}")

    with col5:
        pos_pct = (filtered_df['LSTM_SENTIMENT'] == 'Positive').sum() / len(filtered_df) * 100
        st.metric("Positive %", f"{pos_pct:.1f}%")

    st.markdown("---")

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sentiment Distribution")
        sent_counts = filtered_df['LSTM_SENTIMENT'].value_counts()
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            title='Tweet Sentiment Breakdown',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Return Distribution")
        fig = px.histogram(
            filtered_df,
            x='1_DAY_RETURN',
            nbins=60,
            title='1-Day Stock Returns',
            color_discrete_sequence=['#667eea']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Stock performance table
    st.subheader("🏆 Top 10 Stocks by Average Return")

    stock_perf = filtered_df.groupby('STOCK').agg({
        '1_DAY_RETURN': ['mean', 'std', 'count'],
        'VOLATILITY_10D': 'mean',
        'LSTM_POLARITY': 'mean'
    }).round(4)
    stock_perf.columns = ['Avg_Return', 'Std_Return', 'Count', 'Avg_Volatility', 'Avg_Sentiment']
    stock_perf = stock_perf.sort_values('Avg_Return', ascending=False).head(10).reset_index()

    st.dataframe(
        stock_perf.style.background_gradient(cmap='RdYlGn', subset=['Avg_Return']),
        use_container_width=True
    )

    # Time series
    st.subheader("📅 Tweet Activity Timeline")

    daily_activity = filtered_df.groupby(filtered_df['DATE'].dt.date).size().reset_index()
    daily_activity.columns = ['Date', 'Tweet_Count']

    fig = px.area(
        daily_activity,
        x='Date',
        y='Tweet_Count',
        title='Daily Tweet Volume',
        color_discrete_sequence=['#667eea']
    )
    st.plotly_chart(fig, use_container_width=True)

# =======================
# PAGE 2: DATA EXPLORER
# =======================
elif page == "📊 Data Explorer":
    st.title("📊 Interactive Data Exploration")

    tab1, tab2, tab3 = st.tabs(["🔍 Scatter Analysis", "📦 Distributions", "🔥 Correlations"])

    with tab1:
        st.subheader("Scatter Plot Analysis")

        numeric_cols = [
            'LAST_PRICE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN',
            '7_DAY_RETURN', 'PX_VOLUME_LOG', 'VOLATILITY_10D', 'VOLATILITY_30D',
            'LSTM_POLARITY', 'TEXTBLOB_POLARITY'
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            x_var = st.selectbox("X-axis", numeric_cols, index=8)
        with col2:
            y_var = st.selectbox("Y-axis", numeric_cols, index=1)
        with col3:
            color_by = st.selectbox("Color By", ['LSTM_SENTIMENT', 'STOCK'])

        sample_size = st.slider("Sample Size", 100, min(5000, len(filtered_df)), 2000)
        sample_df = filtered_df.sample(n=min(sample_size, len(filtered_df)))

        fig = px.scatter(
            sample_df,
            x=x_var,
            y=y_var,
            color=color_by,
            title=f'{x_var} vs {y_var}',
            trendline="ols",
            hover_data=['STOCK', 'DATE'],
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        # Calculate correlation
        corr = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
        st.info(f"**Correlation:** {corr:.4f}")

    with tab2:
        st.subheader("Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            cat_var = st.selectbox("Category", ['LSTM_SENTIMENT', 'TEXTBLOB_SENTIMENT'])
        with col2:
            num_var = st.selectbox("Numeric Variable", numeric_cols, index=1)

        fig = px.box(
            filtered_df,
            x=cat_var,
            y=num_var,
            color=cat_var,
            title=f'{num_var} Distribution by {cat_var}',
            points="outliers"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add violin plot
        fig2 = px.violin(
            filtered_df,
            x=cat_var,
            y=num_var,
            color=cat_var,
            title=f'{num_var} Violin Plot by {cat_var}',
            box=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Correlation Analysis")

        corr_matrix = filtered_df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Positive Correlations:**")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False).head(5)
            st.dataframe(corr_df, use_container_width=True)

        with col2:
            st.markdown("**Correlations with 1-Day Return:**")
            target_corr = corr_matrix['1_DAY_RETURN'].sort_values(ascending=False)
            st.dataframe(target_corr, use_container_width=True)

# =======================
# PAGE 3: PREDICTIONS
# =======================
elif page == "🤖 Predictions":
    st.title("🤖 Stock Return Predictions")
    st.markdown("Use trained Random Forest models to predict stock returns")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Input Features")

        user_inputs = {}

        for feature in best_features:
            if feature in filtered_df.columns:
                mean_val = filtered_df[feature].mean()
                std_val = filtered_df[feature].std()
                min_val = filtered_df[feature].min()
                max_val = filtered_df[feature].max()

                user_inputs[feature] = st.number_input(
                    f"**{feature}**",
                    value=float(mean_val),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    step=float(std_val/20),
                    format="%.6f",
                    help=f"Mean: {mean_val:.4f}, Std: {std_val:.4f}"
                )

        predict_btn = st.button("🎯 Make Prediction", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Results")

        if predict_btn:
            # Create input dataframe
            input_df = pd.DataFrame([user_inputs])

            # Make predictions
            reg_pred = models['regressor'].predict(input_df)[0]
            class_pred = models['classifier'].predict(input_df)[0]
            class_proba = models['classifier'].predict_proba(input_df)[0]

            # Display results
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric(
                    "Predicted Return",
                    f"{reg_pred*100:.3f}%",
                    delta="Regression"
                )

            with col_b:
                color = "green" if class_pred == "Positive" else ("red" if class_pred == "Negative" else "gray")
                st.markdown(f"**Return Category**")
                st.markdown(f"<h2 style='color:{color}'>{class_pred}</h2>", unsafe_allow_html=True)

            with col_c:
                confidence = max(class_proba) * 100
                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )

            # Probability chart
            st.markdown("**Class Probabilities:**")
            prob_df = pd.DataFrame({
                'Class': ['Negative', 'Neutral', 'Positive'],
                'Probability': class_proba
            })

            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                color='Probability',
                color_continuous_scale='RdYlGn',
                title='Prediction Confidence Distribution'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Feature comparison
            st.markdown("**Feature Value Comparison:**")
            comp_df = pd.DataFrame({
                'Feature': best_features,
                'Your Input': [user_inputs[f] for f in best_features],
                'Dataset Mean': [filtered_df[f].mean() for f in best_features],
                'Difference (%)': [(user_inputs[f] - filtered_df[f].mean()) / filtered_df[f].mean() * 100 for f in best_features]
            })
            st.dataframe(
                comp_df.style.background_gradient(cmap='RdYlGn', subset=['Difference (%)']),
                use_container_width=True
            )

        else:
            st.info("👈 Enter feature values and click 'Make Prediction'")

            # Show example
            st.markdown("**Example: Random Sample from Dataset**")
            example = filtered_df[best_features].sample(1)
            example_pred = models['regressor'].predict(example)[0]

            st.write(f"**Predicted Return:** {example_pred*100:.3f}%")
            st.dataframe(example, use_container_width=True)

# =======================
# PAGE 4: MODEL PERFORMANCE
# =======================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Regression", "🎯 Classification", "📉 Feature Analysis"])

    with tab1:
        st.subheader("Regression Model Comparison")

        reg_results = data['regression_results']

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                reg_results,
                x='Model',
                y='R2_Score',
                title='R² Score (Higher is Better)',
                color='R2_Score',
                color_continuous_scale='Viridis',
                text='R2_Score'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                reg_results,
                x='Model',
                y='RMSE',
                title='RMSE (Lower is Better)',
                color='RMSE',
                color_continuous_scale='Reds_r',
                text='RMSE'
            )
            fig.update_traces(texttemplate='%{text:.6f}', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Best model
        best_idx = reg_results['R2_Score'].idxmax()
        best = reg_results.loc[best_idx]

        st.success(f"🏆 **Best Model:** {best['Model']} with R² = {best['R2_Score']:.6f}")

        st.dataframe(
            reg_results.style.highlight_max(subset=['R2_Score'], color='lightgreen')
                           .highlight_min(subset=['RMSE'], color='lightgreen'),
            use_container_width=True
        )

    with tab2:
        st.subheader("Classification Model Comparison")

        class_results = data['classification_results']

        fig = px.bar(
            class_results,
            x='Model',
            y='Accuracy',
            title='Classification Accuracy',
            color='Accuracy',
            color_continuous_scale='Blues',
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.add_hline(y=0.333, line_dash="dash", line_color="red",
                     annotation_text="Random Baseline (33.3%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Best model
        best_idx = class_results['Accuracy'].idxmax()
        best = class_results.loc[best_idx]

        st.success(f"🏆 **Best Model:** {best['Model']} with Accuracy = {best['Accuracy']:.6f}")

        st.dataframe(
            class_results.style.highlight_max(subset=['Accuracy'], color='lightblue'),
            use_container_width=True
        )

    with tab3:
        st.subheader("Feature Importance & PCA")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Random Forest Feature Importance**")
            fig = px.bar(
                data['feature_importance'],
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Plasma'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**PCA Variance Explained**")
            pca_info = data['pca_info']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pca_info['Component'],
                y=pca_info['Cumulative_Variance'],
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='#667eea', width=3)
            ))
            fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                         annotation_text="80% Threshold")
            fig.update_layout(
                title='Cumulative Variance Explained',
                xaxis_title='Principal Component',
                yaxis_title='Cumulative Variance'
            )
            st.plotly_chart(fig, use_container_width=True)

        n_comp_80 = (pca_info['Cumulative_Variance'] >= 0.8).idxmax() + 1
        st.info(f"**{n_comp_80} components** explain 80% of variance")

# =======================
# PAGE 5: KEY INSIGHTS
# =======================
elif page == "💡 Key Insights":
    st.title("💡 Key Insights & Findings")

    # Main finding
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 0rem 1rem;
        }

        /* Metrics styling */
        .stMetric {
            background: linear-gradient(135deg, #1e1e2f 0%, #2c2c54 100%);
            padding: 15px;
            border-radius: 10px;
            color: #f1f1f1 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.4);
        }
        .stMetric label {
            color: #f1f1f1 !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #f1f1f1 !important;
        }

        /* Headers */
        h1, h2, h3, h4 {
            background: linear-gradient(135deg, #b3cfff 0%, #6a8eff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }

        /* Insight boxes (dark theme) */
        .insight-box {
            background-color: #1e1e2f;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #6a8eff;
            margin: 10px 0;
            color: #f0f0f0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }

        /* Success box (dark green tint) */
        .success-box {
            background-color: #1d2b1d;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #28a745;
            color: #d0ffd0;
            box-shadow: 0 3px 8px rgba(0,0,0,0.4);
        }

        /* Warning box (dark amber tint) */
        .warning-box {
            background-color: #2d2415;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #ffc107;
            color: #ffeab5;
            box-shadow: 0 3px 8px rgba(0,0,0,0.4);
        }

        /* Make Streamlit tabs dark-friendly */
        div[data-baseweb="tab-list"] {
            background-color: #1b1b2e !important;
            border-radius: 8px;
        }
    .success-box h4, 
    .warning-box h4, 
    .insight-box h4 {
        background: none !important;
        -webkit-text-fill-color: #ffffff !important;
        color: #ffffff !important;
        </style>
    """, unsafe_allow_html=True)
    


    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_r2 = data['regression_results']['R2_Score'].max()
        st.metric("Best R²", f"{best_r2:.4f}")

    with col2:
        best_acc = data['classification_results']['Accuracy'].max()
        st.metric("Best Accuracy", f"{best_acc:.4f}")

    with col3:
        top_corr = data['correlation'].max().values[0]
        st.metric("Max Correlation", f"{top_corr:.4f}")

    with col4:
        n_features = len(best_features)
        st.metric("Best Features", str(n_features))

    # Two column insights
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ What Works</h4>
            <ul>
                <li><strong>Random Forest</strong> performs best (R² = 0.68)</li>
                <li>Volatility is strongest predictor</li>
                <li>Non-linear relationships dominate</li>
                <li>Ensemble methods outperform linear models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Limitations</h4>
            <ul>
                <li>Sentiment alone: weak predictor (r = 0.06-0.12)</li>
                <li>Dataset: few stocks dominate</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.subheader("📊 Key Visualizations")

    tab1, tab2 = st.tabs(["Sentiment vs Returns", "Feature Rankings"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.box(
                filtered_df,
                x='LSTM_SENTIMENT',
                y='1_DAY_RETURN',
                color='LSTM_SENTIMENT',
                title='Returns by Sentiment',
                color_discrete_map={
                    'Positive': 'green',
                    'Negative': 'red',
                    'Neutral': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            sample = filtered_df.sample(min(3000, len(filtered_df)))
            fig = px.scatter(
                sample,
                x='LSTM_POLARITY',
                y='1_DAY_RETURN',
                trendline='ols',
                title='Sentiment vs Returns (by Volatility)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            data['feature_importance'],
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Ranking',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.subheader("🚀 Recommendations")

    st.markdown("""
    <div class="insight-box">
        <h4>Future Improvements</h4>
        <ol>
            <li><strong>Data Enhancement:</strong> Add tweet metadata (retweets, user followers)</li>
            <li><strong>Advanced Models:</strong> Try FinBERT for financial sentiment</li>
            <li><strong>Real-time:</strong> Implement live tweet streaming</li>
            <li><strong>Risk Management:</strong> Develop volatility-adjusted strategies</li>
            <li><strong>Broader Coverage:</strong> Include more stocks and longer periods</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =======================
# FOOTER
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Tweet Sentiment Stock Analysis Dashboard</strong></p>
    <p>Foundations of Data Science Project | October 2025</p>
    <p>Built with Streamlit, Plotly & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
