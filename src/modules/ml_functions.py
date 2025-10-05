"""
Funciones de Machine Learning y Análisis de Tránsito (sin Flask)
Para usar directamente en Streamlit
"""
import pandas as pd
import numpy as np
from scipy.signal import lombscargle
from src.utils import detect_dataset
from src.Global import TARGET_COLUMNS


def normalize_target(series):
    """Normalize target column values (robusto: text codes + numeric '0'/'1')."""
    s = series.copy().fillna('').astype(str).str.upper().str.strip()

    s = s.replace({
        'CONFIRM': 'CONFIRMED',
        'CONFIRMED PLANET': 'CONFIRMED',
        'FALSE_POSITIVE': 'FALSE POSITIVE',
        'FALSEPOSITIVE': 'FALSE POSITIVE',
        'FP': 'FALSE POSITIVE',
        'PC': 'CANDIDATE',
        'CP': 'CANDIDATE',
        'K2 CANDIDATE': 'CANDIDATE',
        # Map numeric representations to readable labels (UPPER CASE)
        '0': 'NOT CANDIDATE',
        '1': 'CANDIDATE',
        '0.0': 'NOT CANDIDATE',
        '1.0': 'CANDIDATE'
    })
    return s


def remove_outliers_iqr(data, column):
    """Remove outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2.0 * IQR
    upper = Q3 + 2.0 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]


def get_class_distribution(df):
    """Get class distribution for train/test split"""
    dataset_type = detect_dataset(df)
    target_col = TARGET_COLUMNS.get(dataset_type, None) if dataset_type else None
    
    if not target_col:
        for col in df.columns:
            if 'disposition' in col.lower():
                target_col = col
                break
    
    if not target_col or target_col not in df.columns:
        return {
            'error': 'Target column not found',
            'available_columns': list(df.columns)
        }
    
    # Contar clases (asegurándanos de que sean int)
    value_counts = df[target_col].value_counts()
    
    # Mapear SIEMPRE a etiquetas legibles
    label_map = {0: "Not Candidate", 1: "Candidate"}
    class_counts = {}
    
    for value, count in value_counts.items():
        # Convertir a int si es posible
        try:
            int_value = int(value)
            readable_label = label_map.get(int_value, str(value))
            class_counts[readable_label] = int(count)
        except:
            class_counts[str(value)] = int(count)
    
    total = len(df)
    train_size = int(total * 0.8)
    
    return {
        'train': class_counts,
        'test': class_counts,
        'total_samples': total,
        'train_size': train_size,
        'test_size': total - train_size,
        'target_column_used': target_col
    }


def transit_analysis(df):
    """Transit analysis - Universal (KOI, TOI, K2)"""
    dataset_type = detect_dataset(df)
    target_col = TARGET_COLUMNS.get(dataset_type, None) if dataset_type else None

    if not target_col:
        for col in df.columns:
            if 'disposition' in col.lower():
                target_col = col
                break

    if not target_col or target_col not in df.columns:
        return {'error': 'Cannot detect target column'}

    duration_col = None
    depth_col = None

    # BUSCAR COLUMNAS EXISTENTES
    for col in df.columns:
        col_lower = col.lower()
        if duration_col is None and any(k in col_lower for k in [
            'duration', 'trandur', 'pl_trandurh', 'koi_duration'
        ]):
            duration_col = col
        if depth_col is None and any(k in col_lower for k in [
            'depth', 'trandep', 'pl_trandep', 'koi_depth', 'ppm'
        ]):
            depth_col = col

    # CREAR COLUMNAS DERIVADAS SI NO EXISTEN
    if not duration_col:
        if 'koi_period' in df.columns:
            df['derived_duration'] = df['koi_period'] / 100
            duration_col = 'derived_duration'
        elif 'pl_orbper' in df.columns:
            df['derived_duration'] = df['pl_orbper'] / 100
            duration_col = 'derived_duration'

    if not depth_col:
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            df['derived_depth'] = (df['koi_prad'] / df['koi_srad']) ** 2 * 10000
            depth_col = 'derived_depth'
        elif 'pl_rade' in df.columns and 'st_rad' in df.columns:
            df['derived_depth'] = (df['pl_rade'] / df['st_rad']) ** 2 * 10000
            depth_col = 'derived_depth'

    # VERIFICAR DESPUÉS DE INTENTAR CREAR
    if not duration_col or not depth_col:
        return {
            'warning': 'No transit columns found',
            'available_columns': list(df.columns)[:20],
            'duration_found': duration_col is not None,
            'depth_found': depth_col is not None
        }

    # FILTRAR DATOS VÁLIDOS
    valid_data = df[[duration_col, depth_col, target_col]].copy()
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if valid_data.empty:
        return {
            'warning': 'No valid data after filtering',
            'duration_column': duration_col,
            'depth_column': depth_col
        }

    # NORMALIZAR TARGET
    valid_data[target_col] = normalize_target(valid_data[target_col])

    # LIMITAR PUNTOS
    if len(valid_data) > 5000:
        valid_data = valid_data.sample(5000, random_state=42)

    return {
        'duration': valid_data[duration_col].tolist(),
        'depth': valid_data[depth_col].tolist(),
        'disposition': valid_data[target_col].tolist(),
        'duration_column': duration_col,
        'depth_column': depth_col,
        'total_points': len(valid_data)
    }


def transit_analysis_no_outliers(df):
    """Transit analysis without outliers"""
    dataset_type = detect_dataset(df)
    target_col = TARGET_COLUMNS.get(dataset_type, None) if dataset_type else None

    # Buscar target
    if not target_col:
        for col in df.columns:
            if 'disposition' in col.lower():
                target_col = col
                break

    if not target_col or target_col not in df.columns:
        return {'error': 'Cannot detect target column'}

    # BÚSQUEDA MEJORADA de columnas duration y depth
    duration_col = None
    depth_col = None

    for col in df.columns:
        col_lower = col.lower()
        
        if duration_col is None and any(k in col_lower for k in [
            'duration', 'trandur', 'pl_trandurh', 
            'koi_duration', 'k2_trandur', 'k2_duration'
        ]):
            duration_col = col
        
        if depth_col is None and any(k in col_lower for k in [
            'depth', 'trandep', 'pl_trandep', 
            'koi_depth', 'k2_trandep', 'k2_depth', 'ppm'
        ]):
            depth_col = col

    # Si no encuentra, intentar derivar
    if not duration_col:
        if 'koi_period' in df.columns:
            df['derived_duration'] = df['koi_period'] / 100
            duration_col = 'derived_duration'
        elif 'pl_orbper' in df.columns:
            df['derived_duration'] = df['pl_orbper'] / 100
            duration_col = 'derived_duration'

    if not depth_col:
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            df['derived_depth'] = (df['koi_prad'] / df['koi_srad']) ** 2 * 10000
            depth_col = 'derived_depth'
        elif 'pl_rade' in df.columns and 'st_rad' in df.columns:
            df['derived_depth'] = (df['pl_rade'] / df['st_rad']) ** 2 * 10000
            depth_col = 'derived_depth'

    if not duration_col or not depth_col:
        return {
            'warning': 'No transit columns found',
            'hint': f'Looking for duration/depth. Available: {list(df.columns)[:20]}',
            'duration_col_found': duration_col,
            'depth_col_found': depth_col
        }

    # Filtrar datos válidos
    valid_data = df[[duration_col, depth_col, target_col]].copy()
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if valid_data.empty:
        return {'warning': 'No valid data after filtering nulls'}

    # Normalizar target
    valid_data[target_col] = normalize_target(valid_data[target_col])

    # Remover outliers con umbral más suave (1.5 IQR en lugar de 2.0)
    def remove_outliers_soft(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]

    cleaned_data = remove_outliers_soft(valid_data, duration_col)
    cleaned_data = remove_outliers_soft(cleaned_data, depth_col)

    if cleaned_data.empty:
        return {
            'warning': 'All data filtered as outliers',
            'hint': 'Try using transit_analysis without outlier removal',
            'original_count': len(valid_data)
        }

    if len(cleaned_data) > 5000:
        cleaned_data = cleaned_data.sample(5000, random_state=42)

    # Estadísticas por disposición
    unique_dispositions = cleaned_data[target_col].unique()
    stats_by_disposition = {}
    
    for disp in unique_dispositions:
        subset = cleaned_data[cleaned_data[target_col] == disp]
        stats_by_disposition[disp] = {
            'count': int(len(subset)),
            'duration_mean': float(subset[duration_col].mean()),
            'depth_mean': float(subset[depth_col].mean())
        }

    return {
        'duration': cleaned_data[duration_col].tolist(),
        'depth': cleaned_data[depth_col].tolist(),
        'disposition': cleaned_data[target_col].tolist(),
        'duration_column': duration_col,
        'depth_column': depth_col,
        'unique_dispositions': list(unique_dispositions),
        'stats_by_disposition': stats_by_disposition,
        'original_count': int(len(df)),
        'cleaned_count': int(len(cleaned_data)),
        'outliers_removed': int(len(valid_data) - len(cleaned_data)),
        'outliers_percentage': float((len(valid_data) - len(cleaned_data)) / len(valid_data) * 100) if len(valid_data) > 0 else 0
    }


def lightcurve_simulation():
    """Simulate light curves for different exoplanet types"""
    def simulate_transit(time, period, depth, duration):
        phase = (time % period) / period
        flux = np.ones_like(time)
        transit_phase = duration / period / 2
        in_transit = np.abs(phase - 0.5) < transit_phase
        flux[in_transit] = 1 - depth
        flux += np.random.normal(0, 0.001, len(time))
        return flux
    
    time = np.linspace(0, 30, 500)
    curves = [
        {'time': time.tolist(), 'flux': simulate_transit(time, period=3.5, depth=0.01, duration=0.2).tolist(), 'label': 'Hot Jupiter (P=3.5d)'},
        {'time': time.tolist(), 'flux': simulate_transit(time, period=10.0, depth=0.003, duration=0.15).tolist(), 'label': 'Super-Earth (P=10d)'},
        {'time': time.tolist(), 'flux': simulate_transit(time, period=20.0, depth=0.0008, duration=0.25).tolist(), 'label': 'Earth-like (P=20d)'}
    ]
    return {'curves': curves}


def phase_folded(df):
    """Generate a phase-folded light curve"""
    period_col = None
    for col in df.columns:
        if 'period' in col.lower():
            period_col = col
            break
    
    if period_col:
        dataset_type = detect_dataset(df)
        target_col = TARGET_COLUMNS.get(dataset_type, None)
        
        if target_col and target_col in df.columns:
            confirmed = df[df[target_col].str.upper() == 'CONFIRMED'].dropna(subset=[period_col])
            
            if len(confirmed) > 0:
                period = confirmed[period_col].iloc[0]
                time = np.linspace(0, period * 3, 300)
                phase = (time % period) / period
                depth = 0.01
                duration = 0.1
                flux = np.ones_like(phase)
                in_transit = np.abs(phase - 0.5) < duration / 2
                flux[in_transit] = 1 - depth
                flux += np.random.normal(0, 0.002, len(flux))
                sort_idx = np.argsort(phase)
                
                return {
                    'phase': phase[sort_idx].tolist(),
                    'flux': flux[sort_idx].tolist(),
                    'period': float(period)
                }
    
    return {'error': 'Could not generate phase folded curve'}


def periodogram():
    """Generate a Lomb-Scargle periodogram"""
    np.random.seed(42)
    time = np.sort(np.random.uniform(0, 100, 200))
    true_period = 5.3
    signal = np.sin(2 * np.pi * time / true_period)
    noise = np.random.normal(0, 0.5, len(time))
    flux = signal + noise
    frequencies = np.linspace(0.01, 1, 1000)
    power = lombscargle(time, flux, 2 * np.pi * frequencies, normalize=True)
    
    return {
        'frequency': frequencies.tolist(),
        'power': power.tolist()
    }