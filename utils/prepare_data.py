'''The script does:
    1. Read csv file and removes unnecessary headers.
    2. Scale overall data to 0-99 range and make it integer. 
    3. Split data into train, test, and validation sets
     
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def prepare_data(file_path, k):
    # Read csv file
    df_ftir = pd.read_csv(file_path, header=None)
    
    # Remove unnecessary headers
    wavenumber = df_ftir.iloc[0,1:].to_numpy()
    time = df_ftir.iloc[1:, 0].to_numpy().astype(float)
    df = df_ftir.iloc[1:, 1:].to_numpy()

    # Convert to Absorbance from Transmittance
    # df = np.log10(np.max(df))-np.log10(df)

    # Row-wise min-max scaling
    scaler = MinMaxScaler(feature_range=(0, 99))
    scaled_data = np.apply_along_axis(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten(), 1, df)
    # scaled_data = scaled_data.round().astype(int)  # Convert to integers

    # Create target data that is k length shifted forward than the source data
    target_data = np.roll(scaled_data, -k, axis=0)[:-k]

    # Adjust source data to match the length of target data
    source_data = scaled_data[:-k]

    # Split data into train, test, and validation sets
    train_data = source_data
    train_target = target_data
    
    val_data, test_data, val_target, test_target = train_test_split(source_data, target_data, test_size=0.5, random_state=42,shuffle=True)
    
    return train_data, val_data, test_data, train_target, val_target, test_target, (wavenumber,time)

