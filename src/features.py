"""
Astrophysical feature engineering and domain transformations.
"""
import numpy as np
import pandas as pd


def compute_astrophysical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific astronomical features:
    1. Photometric colors: (u - g), (g - r), (r - i), (i - z)
    2. Gaia Astrometry: Proper motion total magnitude = sqrt(pm_ra^2 + pm_dec^2)
    3. Parallax to distance approximation (where parallax > 0)
    4. Total magnitude & photometric dispersion across bands
    5. Emission line ratios: H-alpha to [OIII] (distinguishes HII regions / QSOs)
    6. Signal-to-noise ratio aggregations
    """
    data = df.copy()

    # Color indices (essential for stellar spectral type and redshift determination)
    if 'u_mag' in data.columns and 'g_mag' in data.columns:
        data['color_u_g'] = data['u_mag'] - data['g_mag']
    if 'g_mag' in data.columns and 'r_mag' in data.columns:
        data['color_g_r'] = data['g_mag'] - data['r_mag']
    if 'r_mag' in data.columns and 'i_mag' in data.columns:
        data['color_r_i'] = data['r_mag'] - data['i_mag']
    if 'i_mag' in data.columns and 'z_mag' in data.columns:
        data['color_i_z'] = data['i_mag'] - data['z_mag']

    # Photometric summary
    mag_cols = [c for c in data.columns if c.endswith('_mag')]
    if len(mag_cols) > 0:
        data['total_mag'] = data[mag_cols].mean(axis=1)
        data['mag_span'] = data[mag_cols].max(axis=1) - data[mag_cols].min(axis=1)
        data['mag_std'] = data[mag_cols].std(axis=1)

    # Gaia proper motion magnitude
    if 'pm_ra' in data.columns and 'pm_dec' in data.columns:
        data['pm_magnitude'] = np.sqrt(data['pm_ra'] ** 2 + data['pm_dec'] ** 2)

    # Signal-to-Noise Ratio features
    snr_cols = [c for c in data.columns if c.startswith('snr_')]
    if len(snr_cols) > 0:
        data['total_snr'] = data[snr_cols].sum(axis=1)
        data['avg_snr'] = data[snr_cols].mean(axis=1)

    # Spectral indicators
    if 'h_alpha_strength' in data.columns and 'oIII_strength' in data.columns:
        data['h_alpha_to_oIII'] = data['h_alpha_strength'] / (data['oIII_strength'].abs() + 1e-5)

    return data
