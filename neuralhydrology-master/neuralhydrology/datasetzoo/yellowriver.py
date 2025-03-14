from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class YellowRiver(BaseDataset):
    """Template class for adding a new data set.
    
    Each dataset class has to derive from `BaseDataset`, which implements most of the logic for preprocessing data and 
    preparing data for model training. Only two methods have to be implemented for each specific dataset class: 
    `_load_basin_data()`, which loads the time series data for a single basin, and `_load_attributes()`, which loads 
    the static attributes for the specific data set. 
    
    Usually, we outsource the functions to load the time series and attribute data into separate functions (in the
    same file), which we then call from the corresponding class methods. This way, we can also use specific basin data
    or dataset attributes without these classes.
    
    To make this dataset available for model training, don't forget to add it to the `get_dataset()` function in 
    'neuralhydrology.datasetzoo.__init__.py'

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).

    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(YellowRiver, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data
        
        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.
        
        Parameters
        ----------
        basin : str
            Basin identifier as string.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        # for forcing in self.cfg.forcings:
        #     df, area = load_camels_us_forcings(self.cfg.data_dir, basin, forcing)
        #
        #     # rename columns
        #     if len(self.cfg.forcings) > 1:
        #         df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
        #     dfs.append(df)
        # df = pd.concat(dfs, axis=1)

        # add discharge
        df= load_yellow_river_discharge(self.cfg.data_dir, basin)



        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes
        
        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
        dataframe with features in columns.
        
        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        ########################################
        # Add your code for data loading here. #
        ########################################
        pass


def load_yellow_river_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """
    Load the discharge data for a specific basin and standardize the qobs column.

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the station CSV files.
    basin : str
        Identifier of the basin (e.g., "1" for ID_1.csv).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date' as DatetimeIndex and standardized qobs column.
    """
    # Construct the file path for the given basin
    file_path = data_dir / f"ID_{basin}.csv"

    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"No data found for basin: {basin} at {file_path}")

    # Load the data
    df = pd.read_csv(file_path)

    # Check if required columns are present
    if 'time' not in df.columns or 'qobs' not in df.columns:
        raise ValueError(f"The file {file_path} must contain 'time' and 'qobs' columns.")

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Set 'time' column as the index
    df.set_index('time', inplace=True)

    # Ensure the index is a DatetimeIndex and set the name to 'date'
    df.index = pd.DatetimeIndex(df.index, name='date')

    # Ensure the frequency of the index is 'D'
    if df.index.freq is None:
        df.index = df.asfreq('D').index

    # Standardize the qobs column using z-score (mean and standard deviation)
    qobs_mean = df['qobs'].mean()
    qobs_std = df['qobs'].std()

    if qobs_std == 0:
        raise ValueError(f"Standard deviation of qobs in {file_path} is zero, cannot standardize.")

    df['qobs'] = (df['qobs'] - qobs_mean) / qobs_std

    return df

