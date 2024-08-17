import datetime as dt
import re
from pathlib import Path

# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATA_FOLDER = \
    (Path(__file__).parent / "../../../data/copernicus/monthly/2m-temperature").resolve()


def filter_by_time_range(res: pd.DataFrame, time_range: pd.DatetimeIndex):
    
    time_level = res.index.get_level_values("time")
    
    # Create the mask
    mask = time_level.isin(time_range)
    
    # Apply the mask
    res = res[mask]
    
    return res


def get_data(
    start_month_year: tuple[int, int], 
    end_month_year: tuple[int, int],
    longitude: float,
    latitude: float) -> pd.Series:
    
    ## get a list of which data files to access 
    all_data_paths = list(DATA_FOLDER.glob("*.tsv.zip"))
    batch_paths = list()
    year_pattern = r"\d{4}"
    year_range = range(start_month_year[1], end_month_year[1] +1)
    for p in all_data_paths:
        matches = re.findall(year_pattern, p.stem)
        if not matches:
            continue
        year = int(matches[0])
        if year in year_range:
            batch_paths.append(p)

    ## create time range for the relevant data pd.data_range
    time_range = pd.date_range(
        start=f"{start_month_year[1]:04d}-{start_month_year[0]:02d}-01",
        end=f"{end_month_year[1]:04d}-{end_month_year[0]:02d}-01",
        freq="MS"
    )

    ## adjust longitude and latitude parameters to what's available in the data
    adj_lng = int(longitude * 2) / 2
    adj_lat = int(latitude * 2) / 2

    ## iterate through the data files and extract the relevant data 
    res = pd.DataFrame()
    for p in batch_paths:
        df = pd.read_csv(
            str(p), 
            index_col=["longitude", "latitude", "time"],
            parse_dates=["time"],
            sep="\t")

        ## filter the dataframe on adjustet longitude and latitude
        s = df.loc[(adj_lng, adj_lat, )]

        ## concat the filtered data to the result
        res = pd.concat([res, s], axis=0)
        del(df)

    ## filter the result on the timerange
    res = filter_by_time_range(res, time_range)        
    return res


def plot_temperature_with_running_mean(fig, data, window=12, ax=None, **kwargs):
    """
    Plots temperature data with a running mean on the given figure.

    Parameters:
    - fig: matplotlib.figure.Figure object
    - data: pandas Series of temperature data with datetime index
    - window: integer, the window size for computing the running mean
    - ax: matplotlib.axes.Axes object (optional), if None, a new axis will be created
    - **kwargs: additional keyword arguments for customization

    Returns:
    - ax: The axis with the plot
    """
    if ax is None:
        ax = fig.add_subplot(111)

    # Compute the running mean
    running_mean = data.rolling(window=window).mean()

    # Plot the original data
    data.plot(ax=ax, label='Original Data', **kwargs)

    # Plot the running mean
    running_mean.plot(ax=ax, label=f'Running Mean (window={window})', linestyle='--', **kwargs)

    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Data with Running Mean')
    ax.legend()
    
    return ax


## Test
if __name__ == "__main__":
    coord = {
        ## Tromsø, Norway
        #'latitude': 69.66129345159092, 
        #'longitude': 18.939309908268058,
        ## Quito, Ecuador
        #'latitude': -0.5, 
        #'longitude': 360.0 - 78.5,
        ## Oslo, Norway
        ## 'latitude': 60.0, 
        #'longitude': 11.0,
        ## Nuuk, Greenland
        #'latitude': 64.0, 
        #'longitude': 360.0 - 52.0,
        ## Longyearbyen, Svalbard
        #'latitude': 78.0, 
        #'longitude': 15.0,
        ## New York City, USA
        #'latitude': 40.5, 
        #'longitude': 360 - 74.0,
        ## Palma de Mallorca, Spain
        #'latitude': 39.55968237666672, 
        #'longitude': 2.6475542941903,
        ## Yakutsk, Russia
        'latitude': 62.0, 
        'longitude': 130.0,
    }
    temperature_data = get_data(
        start_month_year=(1, 1940), 
        end_month_year=(12, 2023), 
        longitude=coord['longitude'],
        latitude=coord['latitude'])

    fig = plt.figure(figsize=(10, 5))
    ax = plot_temperature_with_running_mean(fig, temperature_data, window=36)
    plt.show()