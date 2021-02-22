# Create functions to featurize our columns.
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
import timeit
import time
from scipy.stats import skew, kurtosis
import concurrent.futures
import multiprocessing

def featurize_column(df, col):
    """Featurizes the dataframe by 10-second windows. Features include mean, std, zero crossing rate, etc.
    Column must be specified (i.e. featurizes from x, y, z, x_tma, y_tma, or z_tma).
    Features are derived from the accelerometer data and not from the TAC columns."""

    def featurize(col):
        df[col + '_mean'] = df[col].rolling('10s').mean()
        df[col + '_std'] = df[col].rolling('10s').std()
        df[col + '_var'] = df[col + '_std'] ** 2
        df[col + '_median'] = df[col].rolling('10s').median()
        df[col + '_max_raw'] = df[col].rolling('10s').max()
        df[col + '_min_raw'] = df[col].rolling('10s').min()
        df[col + '_max_abs'] = (np.absolute(df[col])).rolling('10s').max()
        df[col + '_min_abs'] = (np.absolute(df[col])).rolling('10s').min()
        df[col + '_skew'] = df[col].rolling('10s').skew()
        df[col + '_kurtosis'] = df[col].rolling('10s').apply(lambda a: kurtosis(a))
        # Zero Crossing Rate
        df[col + '_zcr'] = df[col].rolling('10s').apply(lambda a: (np.diff(np.sign(a)) != 0).sum().astype(int))
        # Gait stretch - the difference between max and min of one stride.
        df[col + '_gait_stretch'] = df[col + '_max_raw'] - df[col + '_min_raw']
        # Number of steps - the total number of peaks (maxima) in a window.
        df[col + '_steps'] = df[col + '_max_raw'].rolling('10s').apply(lambda a: (np.diff(a) != 0).sum().astype(int))
        # Step Time - average time between steps.
        df[col + '_step_time'] = 10 / df[col + '_steps']  # 10 second window divided by number of steps
        # Root Mean Squared
        df[col + '_rms'] = df[col].rolling('10s').apply(lambda a: np.sqrt(np.mean(a ** 2)))

        return df

    df[col + '_jerk'] = np.gradient(df[col])
    df[col + '_snap'] = np.gradient(df[col + '_jerk'])

    df = featurize(col)
    df = featurize(col + '_jerk')
    df = featurize(col + '_snap')
    return df


def featurize_multiple_columns(df, col1, col2, col3):
    """Featurizes the dataframe by 10-second windows. Features involve computations along different axes."""

    # Average Resultant Acceleration, calculated using the RMS of each axis
    x = df[col1 + '_rms']
    y = df[col2 + '_rms']
    z = df[col3 + '_rms']
    df[col1 + col2 + col3 + '_ara'] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    x = df[col1 + '_jerk_rms']
    y = df[col2 + '_jerk_rms']
    z = df[col3 + '_jerk_rms']
    df[col1 + col2 + col3 + '_jerk_ara'] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    x = df[col1 + '_snap_rms']
    y = df[col2 + '_snap_rms']
    z = df[col3 + '_snap_rms']
    df[col1 + col2 + col3 + '_snap_ara'] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    df['d' + col1 + '_d' + col2] = df[col1 + '_jerk'] / df[col2 + '_jerk']  # dx/dy
    df['d' + col1 + '_d' + col3] = df[col1 + '_jerk'] / df[col3 + '_jerk']  # dx/dz
    df['d' + col2 + '_d' + col1] = df[col2 + '_jerk'] / df[col1 + '_jerk']  # dy/dx
    df['d' + col2 + '_d' + col3] = df[col2 + '_jerk'] / df[col3 + '_jerk']  # dy/dz
    df['d' + col3 + '_d' + col1] = df[col3 + '_jerk'] / df[col1 + '_jerk']  # dz/dx
    df['d' + col3 + '_d' + col2] = df[col3 + '_jerk'] / df[col2 + '_jerk']  # dz/dy

    df['dd' + col1 + '_dd' + col2] = df[col1 + '_snap'] / df[col2 + '_snap']  # ddx/ddy
    df['dd' + col1 + '_dd' + col3] = df[col1 + '_snap'] / df[col3 + '_snap']  # ddx/ddz
    df['dd' + col2 + '_dd' + col1] = df[col2 + '_snap'] / df[col1 + '_snap']  # ddy/ddx
    df['dd' + col2 + '_dd' + col3] = df[col2 + '_snap'] / df[col3 + '_snap']  # ddy/ddz
    df['dd' + col3 + '_dd' + col1] = df[col3 + '_snap'] / df[col1 + '_snap']  # ddz/ddx
    df['dd' + col3 + '_dd' + col2] = df[col3 + '_snap'] / df[col2 + '_snap']  # ddz/ddy

    return df


def featurize_df(df):
    """Uses the featurize_column function on each accelerometer column (x, y, z)"""
    col1 = 'x'
    col2 = 'y'
    col3 = 'z'
    df = featurize_column(df, col1)
    df = featurize_column(df, col2)
    df = featurize_column(df, col3)
    df = featurize_multiple_columns(df, col1, col2, col3)
    return df


def bifurcate(df):
    """Separates the df into 2 df's:
    One containing the raw x, y, z, and interpolated TAC (3rd, 5th, 7th, and 9th ordered interpolation)
    and the other containing the smoothed weighted averages of x, y, z, and TAC."""

    df_tma = df[['x_tma', 'y_tma', 'z_tma', 'tma']].rename(columns={"x_tma": "x", "y_tma": "y", "z_tma": "z"})
    df = df.drop(columns=['tma', 'x_tma', 'y_tma', 'z_tma'])
    return df_tma, df


def prime_factor(value):
    factors = []
    for divisor in range(2, value-1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
        else:
            factors = [value]
    return factors

def worker(x):
    return x*x


def featurize_std_df(df_name):
    """Takes the name of a csv file, reads it into a pandas dataframe, and sets the "time" column as the index.
    Calculates the standard deviation on all of the features in "col_list", over a 10-second rolling window."""

    start = time.time()

    df = pd.read_csv(df_name)
    df.time = pd.to_datetime(df.time)
    df = df.set_index('time')

    # Note: x, y, z, and _jerk and _snap have been removed to not result in redundant _std features.
    col_list = ['x_mean', \
                'x_std', \
                'x_var', \
                'x_median', \
                'x_max_raw', \
                'x_min_raw', \
                'x_max_abs', \
                'x_min_abs', \
                'x_skew', \
                'x_kurtosis', \
                'x_zcr', \
                'x_gait_stretch', \
                'x_steps', \
                'x_step_time', \
                'x_rms', \
                'x_jerk_mean', \
                'x_jerk_std', \
                'x_jerk_var', \
                'x_jerk_median', \
                'x_jerk_max_raw', \
                'x_jerk_min_raw', \
                'x_jerk_max_abs', \
                'x_jerk_min_abs', \
                'x_jerk_skew', \
                'x_jerk_kurtosis', \
                'x_jerk_zcr', \
                'x_jerk_gait_stretch', \
                'x_jerk_steps', \
                'x_jerk_step_time', \
                'x_jerk_rms', \
                'x_snap_mean', \
                'x_snap_std', \
                'x_snap_var', \
                'x_snap_median', \
                'x_snap_max_raw', \
                'x_snap_min_raw', \
                'x_snap_max_abs', \
                'x_snap_min_abs', \
                'x_snap_skew', \
                'x_snap_kurtosis', \
                'x_snap_zcr', \
                'x_snap_gait_stretch', \
                'x_snap_steps', \
                'x_snap_step_time', \
                'x_snap_rms', \
                'y_mean', \
                'y_std', \
                'y_var', \
                'y_median', \
                'y_max_raw', \
                'y_min_raw', \
                'y_max_abs', \
                'y_min_abs', \
                'y_skew', \
                'y_kurtosis', \
                'y_zcr', \
                'y_gait_stretch', \
                'y_steps', \
                'y_step_time', \
                'y_rms', \
                'y_jerk_mean', \
                'y_jerk_std', \
                'y_jerk_var', \
                'y_jerk_median', \
                'y_jerk_max_raw', \
                'y_jerk_min_raw', \
                'y_jerk_max_abs', \
                'y_jerk_min_abs', \
                'y_jerk_skew', \
                'y_jerk_kurtosis', \
                'y_jerk_zcr', \
                'y_jerk_gait_stretch', \
                'y_jerk_steps', \
                'y_jerk_step_time', \
                'y_jerk_rms', \
                'y_snap_mean', \
                'y_snap_std', \
                'y_snap_var', \
                'y_snap_median', \
                'y_snap_max_raw', \
                'y_snap_min_raw', \
                'y_snap_max_abs', \
                'y_snap_min_abs', \
                'y_snap_skew', \
                'y_snap_kurtosis', \
                'y_snap_zcr', \
                'y_snap_gait_stretch', \
                'y_snap_steps', \
                'y_snap_step_time', \
                'y_snap_rms', \
                'z_mean', \
                'z_std', \
                'z_var', \
                'z_median', \
                'z_max_raw', \
                'z_min_raw', \
                'z_max_abs', \
                'z_min_abs', \
                'z_skew', \
                'z_kurtosis', \
                'z_zcr', \
                'z_gait_stretch', \
                'z_steps', \
                'z_step_time', \
                'z_rms', \
                'z_jerk_mean', \
                'z_jerk_std', \
                'z_jerk_var', \
                'z_jerk_median', \
                'z_jerk_max_raw', \
                'z_jerk_min_raw', \
                'z_jerk_max_abs', \
                'z_jerk_min_abs', \
                'z_jerk_skew', \
                'z_jerk_kurtosis', \
                'z_jerk_zcr', \
                'z_jerk_gait_stretch', \
                'z_jerk_steps', \
                'z_jerk_step_time', \
                'z_jerk_rms', \
                'z_snap_mean', \
                'z_snap_std', \
                'z_snap_var', \
                'z_snap_median', \
                'z_snap_max_raw', \
                'z_snap_min_raw', \
                'z_snap_max_abs', \
                'z_snap_min_abs', \
                'z_snap_skew', \
                'z_snap_kurtosis', \
                'z_snap_zcr', \
                'z_snap_gait_stretch', \
                'z_snap_steps', \
                'z_snap_step_time', \
                'z_snap_rms', \
                'xyz_ara', \
                'xyz_jerk_ara', \
                'xyz_snap_ara', \
                'dx_dy', \
                'dx_dz', \
                'dy_dx', \
                'dy_dz', \
                'dz_dx', \
                'dz_dy', \
                'ddx_ddy', \
                'ddx_ddz', \
                'ddy_ddx', \
                'ddy_ddz', \
                'ddz_ddx', \
                'ddz_ddy']

    for col in col_list:
        df[col + '_std'] = df[col].rolling('10s').std()

    df = df.iloc[::100, :]
    df.to_csv(df_name[:-4] + '_std.csv')  # removes the last 4 characters '.csv' and appends '_std'
    print(df_name, 'Time:', time.time() - start)