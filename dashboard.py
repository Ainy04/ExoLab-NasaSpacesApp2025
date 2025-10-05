"""
dashboard.py - Versión adaptada sin Flask
Usa funciones directas en lugar de API REST
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
from pathlib import Path

# FUnciones locales
from src.modules.data_functions import (
    get_data_info, get_statistics, get_distribution, 
    get_correlation, get_scatter_data, get_sample,
    get_statistics_no_outliers, scatter_no_outliers
)
from src.modules.ml_functions import (
    get_class_distribution, transit_analysis, 
    transit_analysis_no_outliers, lightcurve_simulation,
    phase_folded, periodogram
)
from src.modules.predict_functions import predict
from src.modules.upload_functions import upload_and_train
from src.modules.download_functions import read_model_bytes

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = 'data/uploaded.csv'
MODEL_PATH = 'models/model.joblib'

st.set_page_config(
    page_title="ExoLab Dashboard",
    layout="wide",
    page_icon=str(SCRIPT_DIR / "IconEL.png"),
    initial_sidebar_state="expanded"
)

# Normalización local en frontend: convierte 0/1 (num o str) a etiquetas legibles UPPERCASE
def normalize_disposition_series(s):
    # s: pd.Series
    mapping = {0: 'NOT CANDIDATE', 1: 'CANDIDATE', '0': 'NOT CANDIDATE', '1': 'CANDIDATE', '0.0': 'NOT CANDIDATE', '1.0': 'CANDIDATE'}
    try:
        return s.map(mapping).fillna(s.astype(str).str.upper())
    except Exception:
        return s.astype(str).str.upper()

def render_stats_table(stats_data):
    """Renders statistics table - CORRECTED VERSION"""
    if not stats_data or len(stats_data) == 0:
        return None
    
    # If stats_data is dict of dicts, convert to long format
    if isinstance(stats_data, dict):
        rows = []
        for feature, metrics in stats_data.items():
            if isinstance(metrics, dict):
                row = {'feature': feature}
                row.update(metrics)
                rows.append(row)
            else:
                # Case when it's a pandas series
                rows.append({'feature': feature, 'value': metrics})
        
        df = pd.DataFrame(rows)
    else:
        # If already a DataFrame
        df = pd.DataFrame(stats_data)
    
    return df

def render_correlation_heatmap(correlations, max_features=10):
    """Renders correlation heatmap"""
    if not correlations or len(correlations) == 0:
        return None
    
    # Convert to DataFrame
    df_corr = pd.DataFrame(correlations)
    
    # Limit to first N features
    if len(df_corr) > max_features:
        df_corr = df_corr.iloc[:max_features, :max_features]
    
    return df_corr

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'training_in_progress' not in st.session_state:
    st.session_state['training_in_progress'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = "Data Exploration"


st.markdown("""
<style>
/* Hover effect on metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #394E7B, #4A46A8);
    padding: 15px;
    border-radius: 12px;
    font-weight: bold;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
[data-testid="stMetric"]:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #4A46A8, #6366F1);
    border: none;
    border-radius: 8px;
    transition: all 0.3s ease-in-out;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(74, 70, 168, 0.3);
}
            
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #5B21B6, #7C3AED);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 70, 168, 0.4);
    cursor: pointer;
}

[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(74, 70, 168, 0.3);
}
            
/* General title style */
h1, h2, h3, h4 {
    color: #E8EAF6;
    transition: color 0.3s ease-in-out;
}
h2:hover, h3:hover, h4:hover {
    color: #89BBFE;
}

/* General background */
body {
    background-color: #1E2A47;
}

/* Separators */
hr {
    border: 1px solid #4A46A8;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("ExoLab")

    upload_file = st.file_uploader("Upload your file (CSV)", type=['csv'])
    
    # Selección del modelo
    selected_model = st.radio(
        "Select the ML model to train:",
        ["RandomForest", "XGBoost"],
        index=0,
        horizontal=True
    )

    # Botón de entrenamiento
    if upload_file is not None:
        if st.button("Upload file and Training Model", type="primary"):
            if not st.session_state['training_in_progress']:
                st.session_state['training_in_progress'] = True
                st.rerun()
    
    if st.session_state['training_in_progress']:
        import time

        # Overlay animado (pantalla oscura con spinner)
        overlay = st.empty()
        overlay.markdown("""
        <style>
        .overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .popup {
            background-color: #111827;
            padding: 30px;
            color: white;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            animation: fadeIn 0.3s ease-in-out;
        }
        .spinner {
            border: 6px solid #f3f3f3;
            border-top: 6px solid #89BBFE;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            margin: 0 auto 20px;
            animation: spin 1s linear infinite; 
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        </style>

        <div class="overlay">
            <div class="popup">
                <div class="spinner"></div>
                <h3>Processing data and training the model...</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        start_time = time.time()

        try:
            # LLAMADA DIRECTA A LA FUNCIÓN (SIN FLASK/REQUESTS)
            result = upload_and_train(
                file_content=upload_file,
                selected_model=selected_model,
                data_path=DATA_PATH,
                model_path=MODEL_PATH
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Mostrar ventana de "Training Completed"
            overlay.markdown(f"""
            <style>
            .overlay {{
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background-color: rgba(0, 0, 0, 0.6);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            }}
            .popup {{
                background-color: #23314E;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 0 20px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif;
                animation: fadeIn 0.3s ease-in-out;
                color: White;
            }}
            .spinner {{
                display: none;
            }}
            </style>

            <div class="overlay">
                <div class="popup">
                    <h3>Training Completed!</h3>
                    <p>Total time: {elapsed_time:.2f} seconds</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(4)
            overlay.empty()

            if 'error' not in result:
                st.session_state['training_results'] = result
                st.session_state['model_trained'] = True
                st.session_state['dataset_type'] = result.get('dataset_type')
                st.session_state['page'] = "ML Results"
                st.session_state['training_in_progress'] = False
                
                # Cargar el DataFrame procesado
                st.session_state['df_loaded'] = pd.read_csv(DATA_PATH)
                
                st.success("Model successfully trained.")
                st.rerun()
            else:
                error_msg = result.get('error', 'Unknown Error')
                st.error(f"Error: {error_msg}")
                st.session_state['training_in_progress'] = False

        except Exception as e:
            st.error(f"Training error: {e}")
            overlay.empty()
            st.session_state['training_in_progress'] = False

    st.markdown("---")
    st.markdown("Navigation")

    if st.session_state['training_in_progress']:
        st.warning("Training in progress... Navigation disabled")
        page = st.session_state['page']
    else:
        if not st.session_state['model_trained']:
            page_options = ["Data Exploration"]
            st.info("Train a model to unlock ML Results and Prediction")
        else:
            page_options = ["Data Exploration", "ML Results", "Prediction"]

        page = st.radio(
            "Choose an option:",
            page_options,
            key="nav_radio",
            index=page_options.index(st.session_state['page'])
            if st.session_state['page'] in page_options else 0
        )

        if page != st.session_state['page']:
            st.session_state['page'] = page
            st.rerun()            

# Main Page
# PAGE 1: DATA EXPLORATION (SIN FLASK)
if page == "Data Exploration":
    col1, col2 = st.columns([5, 1])

    with col1:
        st.title("Welcome!")
        st.write("##### Analyze new data and identify new exoplanets.")
        st.markdown("""
            You will be able to view a sample dataset so that you can explore
            the data and test all the dashboard features before uploading
            your own CSV file for machine learning analysis.
        """)

    with col2:
        st.image(str(SCRIPT_DIR / "ExoLab.png"), width=120)

    st.markdown("---")
    st.header("Data Analysis")

    try:
        # Verificar si hay datos cargados
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            info = get_data_info(df)
            dataset_type = info.get('dataset_type', 'unknown')
            
            from data_functions import get_dataset_config
            config_data = get_dataset_config(df)
            available_cols = config_data.get('available_columns', {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Objects", info['total_rows'])
            with col2:
                st.metric("Features", info['total_columns'])
            with col3:
                missing_total = sum(info['missing_values'].values())
                st.metric("Missing values", missing_total)
            with col4:
                st.metric("Dataset Type", dataset_type.upper())

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Raw Data", "Descriptive Statistics", "Feature Distributions", "Preliminary Analysis"
            ])

            with tab1:
                st.subheader("Data")
                n_rows = st.slider("Number of rows:", 5, 100, 10)
                sample_data = get_sample(df, n=n_rows)
                df_sample = pd.DataFrame(sample_data)
                st.dataframe(df_sample, use_container_width=True)

            with tab2:
                st.subheader("Descriptive Statistics")
                stats = get_statistics(df)
                df_stats = pd.DataFrame(stats)
                st.dataframe(df_stats, use_container_width=True)

                st.subheader("Distributions of Features")
                numeric_cols = info['numeric_columns']

                if numeric_cols:
                    selected_feature = st.selectbox("Choose a feature:", numeric_cols)

                    if selected_feature:
                        dist_data = get_distribution(df, selected_feature)

                        if 'error' not in dist_data and dist_data['type'] == 'numeric':
                            fig = go.Figure()
                            bins = dist_data['bins']
                            hist = dist_data['histogram']
                            bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]

                            fig.add_trace(go.Bar(
                                x=bin_centers,
                                y=hist,
                                name=selected_feature,
                                marker_color='#6F8AB7'
                            ))
                            fig.update_layout(
                                title=f"Distribution: {selected_feature}",
                                xaxis_title=selected_feature,
                                yaxis_title="Frequency",
                                height=400,
                                bargap=0.05
                            )

                            st.markdown("---")
                            st.plotly_chart(fig, use_container_width=True)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean", f"{dist_data['mean']:.4f}")
                            with col2:
                                st.metric("Median", f"{dist_data['median']:.4f}")
                            with col3:
                                st.metric("Standard Deviation", f"{dist_data['std']:.4f}")

            with tab3:
                st.subheader("Correlation Matrix")
                numeric_cols = info['numeric_columns']

                if len(numeric_cols) > 1:
                    selected_cols = st.multiselect(
                        "Select features for correlation:",
                        numeric_cols,
                        default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
                    )

                    if len(selected_cols) > 1:
                        corr_data = get_correlation(df, columns=selected_cols)
                        corr_df = pd.DataFrame(corr_data)

                        fig = px.imshow(
                            corr_df,
                            title="Correlation Matrix",
                            color_continuous_scale='Blues',
                            aspect='auto',
                            zmin=-1,
                            zmax=1
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Interactive Scatter Plots")

                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X Axis:", numeric_cols, index=0, key='scatter_x')
                    with col2:
                        y_col = st.selectbox("Y Axis:", numeric_cols, index=min(1, len(numeric_cols)-1), key='scatter_y')

                    color_col = st.selectbox("Color by (optional):", ['None'] + info['columns'])

                    if st.button("Generate Scatter Plot"):
                        scatter_data = get_scatter_data(
                            df, 
                            x_col=x_col, 
                            y_col=y_col, 
                            color_col=color_col if color_col != 'None' else None
                        )

                        if 'error' not in scatter_data and len(scatter_data['x']) > 0:
                            df_scatter = pd.DataFrame({
                                x_col: scatter_data['x'],
                                y_col: scatter_data['y']
                            })

                            if 'color' in scatter_data and len(scatter_data['color']) > 0:
                                df_scatter['color'] = scatter_data['color']
                                df_scatter['color'] = normalize_disposition_series(df_scatter['color'])
                                fig = px.scatter(
                                    df_scatter,
                                    x=x_col,
                                    y=y_col,
                                    color='color',
                                    title=f"{y_col} vs {x_col}",
                                    opacity=0.6
                                )
                            else:
                                fig = px.scatter(
                                    df_scatter,
                                    x=x_col,
                                    y=y_col,
                                    title=f"{y_col} vs {x_col}",
                                    opacity=0.6
                                )

                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                            st.info(f"Showing {len(scatter_data['x'])} valid points")

            with tab4:
                st.subheader("Graphics")
                st.markdown("Preliminary Exploration (Before ML Processing)")

                if available_cols:
                    sample_data_pre = get_sample(df, n=1000)
                    df_pre = pd.DataFrame(sample_data_pre)

                    # Disposition Histogram
                    if 'disposition' in available_cols:
                        disp_col = available_cols['disposition']
                        st.markdown("##### **Disposition Histogram**")

                        df_pre[disp_col] = normalize_disposition_series(df_pre[disp_col])
                        df_disp = df_pre[df_pre[disp_col].isin(['CANDIDATE', 'NOT CANDIDATE'])]

                        color_map = {
                            'CANDIDATE': '#f39c12',
                            'NOT CANDIDATE': '#95a5a6'
                        }

                        fig1 = px.histogram(
                            df_disp,
                            x=disp_col,
                            color=disp_col,
                            title="Distribution of Candidate vs Not Candidate",
                            color_discrete_map=color_map
                        )

                        fig1.update_layout(
                            legend=dict(
                                title="Disposition",
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5
                            ),
                            bargap=0.2,
                            height=400
                        )

                        st.plotly_chart(fig1, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# PAGE 2: ML RESULTS 
elif page == "ML Results":
    if st.session_state['training_in_progress']:
        st.info("Training in progress... Please wait")
        with st.spinner("Training model..."):
            st.stop()

    st.title("Machine Learning Results")
    st.write("##### **Evaluate your trained model's performance.**")
    st.markdown(""" 
        Explore comprehensive metrics, visualizations, and insights about your 
        exoplanet classification model. Analyze feature importance, confusion matrices, 
        and cross-validation results to understand how well your model performs.
    """)
    
    st.markdown("---")

    if 'training_results' not in st.session_state:
        st.warning("First you must train a model in the settings section.")
    else:
        results = st.session_state['training_results']

        st.subheader("Model Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['metrics']['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{results['metrics']['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{results['metrics']['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{results['metrics']['f1']:.2%}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Model Visualizations", "Exoplanet Analysis", "Detailed Metrics"])

        with tab1:
            st.subheader("Model Visualizations")
            
            # Class Distribution
            st.markdown("##### **Class Distribution**")
            
            try:
                if os.path.exists(DATA_PATH):
                    df = pd.read_csv(DATA_PATH)
                    class_dist = get_class_distribution(df)
                    
                    if 'error' not in class_dist:
                        df_dist = pd.DataFrame({
                            'Class': list(class_dist['train'].keys()) * 2,
                            'Count': list(class_dist['train'].values()) + list(class_dist['test'].values()),
                            'Dataset': ['Training'] * len(class_dist['train']) + ['Test'] * len(class_dist['test'])
                        })
                        
                        fig = px.bar(
                            df_dist,
                            x='Class',
                            y='Count',
                            color='Dataset',
                            barmode='group',
                            title="Class Distribution: Training vs. Testing",
                            color_discrete_map={'Training': '#6F8AB7', 'Test': '#4A46A8'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load class distribution: {e}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.markdown("##### **Confusion Matrix**")
            cm = np.array(results['metrics']['confusion_matrix'])
            labels = results.get('label_mapping', ['Not Candidate', 'Candidate'])

            fig = px.imshow(
                cm,
                labels=dict(x="Prediction", y="Actual", color="Count"),
                x=labels,
                y=labels,
                title="Confusion Matrix",
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
            # Feature Importance
            st.markdown("##### **Feature Importance**")
            fi_data = results['feature_importances'][:15]
            df_fi = pd.DataFrame(fi_data)

            fig = px.bar(
                df_fi,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Exoplanet Analysis")
            st.markdown("##### **Transit Duration vs. Depth (Original Data)**")
            st.caption("Complete view of all data without filtering")

            try:
                if os.path.exists(DATA_PATH):
                    df = pd.read_csv(DATA_PATH)
                    transit_data = transit_analysis(df)
                    
                    if 'warning' in transit_data:
                        st.warning(transit_data['warning'])
                        if 'hint' in transit_data:
                            st.info(transit_data['hint'])
                    
                    elif 'duration' in transit_data and len(transit_data['duration']) > 0:
                        df_transit = pd.DataFrame({
                            'duration': transit_data['duration'],
                            'depth': transit_data['depth'],
                            'disposition': transit_data['disposition']
                        })
                        
                        counts = df_transit['disposition'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        for idx, (disp, count) in enumerate(counts.items()):
                            with [col1, col2, col3][idx % 3]:
                                st.metric(disp, count)
                        
                        fig = px.scatter(
                            df_transit,
                            x='duration',
                            y='depth',
                            color='disposition',
                            title=f"Full View - {len(df_transit):,} points",
                            labels={
                                'duration': f"{transit_data.get('duration_column', 'Duration')} (hours)", 
                                'depth': f"{transit_data.get('depth_column', 'Depth')} (ppm)"
                            },
                            opacity=0.5,
                            color_discrete_map={
                                'CONFIRMED': '#2ecc71',
                                'FALSE POSITIVE': '#e74c3c',
                                'CANDIDATE': '#f39c12'
                            },
                            hover_data={
                                'duration': ':.3f',
                                'depth': ':.1f',
                                'disposition': True
                            }
                        )
                        
                        fig.update_traces(marker=dict(size=4))
                        fig.update_layout(
                            height=500,
                            hovermode='closest',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No transit data available")
                        
            except Exception as e:
                st.error(f"Error rendering transit analysis: {e}")
                            
        with tab3:
            st.subheader("Detailed Metrics")
            
            st.markdown("##### **Metrics on Test Set**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{results['metrics']['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{results['metrics']['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{results['metrics']['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{results['metrics']['f1']:.4f}")
            
            st.markdown("---")
            
            # Model Information
            st.markdown("##### **Model Information**")
            
            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:
                st.metric("Training Data", results['train_size'])
                st.metric("Test Data", results['test_size'])

            with info_col2:
                st.metric("Total Rows", results['total_rows'])
                st.metric("Total Features", results['total_features'])
            
            with info_col3:
                test_ratio = results['test_size'] / results['total_rows']
                st.metric("Test/Total Ratio", f"{test_ratio:.2%}")
                labels = results.get('label_mapping', [])
                st.metric("Classes", len(labels))

            # Download Model Button
            if st.button("Download Model"):
                try:
                    model_bytes = read_model_bytes(MODEL_PATH)
                    if model_bytes:
                        st.download_button(
                            label="Download model.joblib",
                            data=model_bytes,
                            file_name="exoplanet_model.joblib",
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("Model file not found")
                except Exception as e:
                    st.error(f"Error: {e}")

# PAGE 3: PREDICTION (SIN FLASK)
elif page == "Prediction":
    if st.session_state['training_in_progress']:
        st.info("Training in progress... Please wait")
        st.stop()

    st.title("Interactive Prediction")
    st.write("##### **Make real-time predictions with your trained model.**")
    st.markdown(""" 
        Input feature values to classify potential exoplanets. The model will 
        analyze your data and predict whether the object is a confirmed exoplanet, 
        a false positive, or a candidate requiring further observation.
    """)
    
    st.markdown("---")
    
    if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
        st.warning("Train a model first.")
    else:
        st.success("Model loaded for prediction.")
        results = st.session_state['training_results']
        features = [f['feature'] for f in results['feature_importances']]

        st.markdown("### Enter Feature Values:")
        input_data = {}
        cols = st.columns(3)

        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(feature, value=0.0, format="%.6f")

        if st.button("Make Prediction", type="primary"):
            try:
                # ===================================================
                # LLAMADA DIRECTA A LA FUNCIÓN (SIN FLASK/REQUESTS)
                # ===================================================
                prediction_result = predict(MODEL_PATH, input_data)
                
                if 'error' not in prediction_result:
                    st.subheader("Prediction Result")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### Prediction: **{prediction_result['prediction']}**")
                        st.markdown(f"**Confidence:** {prediction_result['confidence']:.2%}")
                    with col2:
                        df_prob = pd.DataFrame({
                            'Class': list(prediction_result['probabilities'].keys()),
                            'Probability': list(prediction_result['probabilities'].values())
                        })
                        fig = px.bar(
                            df_prob,
                            x='Class', y='Probability',
                            color='Probability',
                            color_continuous_scale='Viridis',
                            title="Class Probabilities"
                        )
                        fig.update_layout(transition_duration=600, template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Prediction error: {prediction_result['error']}")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; font-size: 13px; line-height: 1.8;'>
        <span style='font-size: 16px;'>🚀</span> 
        <strong style='color: #4a9eff;'>NASA Space Apps Challenge 2025</strong> 
        <span style='color: #888;'>©</span> 
        <strong style='color: #fff;'>ExoLab Dashboard</strong> 
        <span style='color: #666;'>•</span> 
        Desarrollado por <strong style='color: #ff6b6b;'>Astro404</strong> 
        <span style='color: #666;'>•</span> 
        <span style='font-size: 11px; color: #777;'>Todos los derechos reservados</span>
    </div>
    """,
    unsafe_allow_html=True
)