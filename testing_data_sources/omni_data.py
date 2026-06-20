import pandas as pd
import numpy as np
import requests
import io

data_url = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/omni_min202605.asc"

# Hardcoded column names in line with hroformat.txt, Gemini did this part for me
omni_columns = [
    "year", "day", "hour", "minute", "id_imf", "id_sw", 
    "num_pts_imf", "num_pts_sw", "percent_interp", "timeshift", 
    "rms_timeshift", "rms_phase", "time_between_obs", "b_magnitude", 
    "bx_gse", "by_gse", "bz_gse", "by_gsm", "bz_gsm", "rms_b_scalar", 
    "rms_b_vector", "flow_speed", "vx_gse", "vy_gse", "vz_gse", 
    "proton_density", "temperature", "flow_pressure", "e_field", 
    "beta", "mach_number", "x_gse", "y_gse", "z_gse", 
    "bsn_x_gse", "bsn_y_gse", "bsn_z_gse", "ae_index", 
    "al_index", "au_index", "sym_d", "sym_h", "asy_d", 
    "asy_h", "pc_n_index", "magnetosonic_mach", "pr_flux_1", 
    "pr_flux_2", "pr_flux_4", "pr_flux_10", "pr_flux_30", 
    "pr_flux_60", "flux_flag", "al_kp_ap_index", "f107_index"
]

response = requests.get(data_url)
print(f"Request to data source returned status code: {response.status_code}")

df = pd.read_csv(
    io.StringIO(response.text), # read the response content as a string and pass it to StringIO for pandas to read
    sep=r'\s+',
    header=None,
    names=omni_columns,
    on_bad_lines='skip'
)


'''
NASA Uses 999 or 9999 (or similar) as fill values for missing data
so I had Claude read through the hroformat.txt documentation and assign the specific Null values
attributed to each column 
'''
fill_values = {
    # spacecraft / QC metadata (cleaning these is optional; see note below)
    "id_imf": 99, "id_sw": 99,
    "num_pts_imf": 999, "num_pts_sw": 999, "percent_interp": 999,
    "timeshift": 999999, "rms_timeshift": 999999, "time_between_obs": 999999,
    "rms_phase": 99.99,

    # magnetic field, nT  (F8.2 -> 9999.99)
    "b_magnitude": 9999.99,
    "bx_gse": 9999.99, "by_gse": 9999.99, "bz_gse": 9999.99,
    "by_gsm": 9999.99, "bz_gsm": 9999.99,
    "rms_b_scalar": 9999.99, "rms_b_vector": 9999.99,

    # velocity, km/s  (F8.1 -> 99999.9)
    "flow_speed": 99999.9,
    "vx_gse": 99999.9, "vy_gse": 99999.9, "vz_gse": 99999.9,

    # plasma scalars (each a different width, so a different fill)
    "proton_density": 999.99,      # F7.2
    "temperature": 9999999.0,      # F9.0
    "flow_pressure": 99.99,        # F6.2
    "e_field": 999.99,             # F7.2
    "beta": 999.99,                # F7.2
    "mach_number": 999.9,          # F6.1
    "magnetosonic_mach": 99.9,     # F5.1

    # positions, Re  (F8.2 -> 9999.99)
    "x_gse": 9999.99, "y_gse": 9999.99, "z_gse": 9999.99,
    "bsn_x_gse": 9999.99, "bsn_y_gse": 9999.99, "bsn_z_gse": 9999.99,

    # geomagnetic indices, nT  (I6 -> 99999)  <- the integer fill your global replace missed
    "ae_index": 99999, "al_index": 99999, "au_index": 99999,
    "sym_d": 99999, "sym_h": 99999, "asy_d": 99999, "asy_h": 99999,

    # polar cap index  (F7.2 -> 999.99); not populated in recent-year files anyway
    "pc_n_index": 999.99,
}

df.replace({col: {fill: np.nan} for col, fill in fill_values.items()}, inplace=True)

print(f"DataFrame Dimensions: {df.shape[0]} rows x {df.shape[1]} columns.\n")

print(df.groupby('day')[['ae_index', 'sym_h']].count()) 
'''
The groupby was performed to check if there was a legitimate delay in the data because it comes from a specific station in Kyoto
and I needed to see if there were any days with zero observations, 
which would indicate a data gap rather than just a delay in reporting

'''

print(f"\nMissing Values by Column:\n{df.isna().sum()}")
print(df.head())
