import tqdm
import warnings
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import Update_files

#This function creates 4 dataframes, one for each variable.

def create_dataframe():
    #Check if the most recent files have been downloaded
    Update_files.update()
    cwd = os.getcwd()
    regional_data_covid_path = os.path.join(cwd, "regional_data_covid")

    recovered_data = []
    new_daily_infections_data = []
    hospitalized_data = []
    deceased_data = []
    dates = []
    # Load data
    warnings.filterwarnings("ignore")
    for path in glob.glob(regional_data_covid_path):
        for i, subpath in enumerate(tqdm.tqdm(glob.glob(path + os.sep +"*"))):
            # add a new column to each dataframe

            new_df = pd.read_csv(subpath) #, index_col=0
            dates.append(new_df.data[0])
            recovered_data.append(np.array(new_df.dimessi_guariti))
            new_daily_infections_data.append(np.array(new_df.nuovi_positivi))
            hospitalized_data.append(np.array(new_df.totale_ospedalizzati))
            deceased_data.append(np.array(new_df.deceduti))
    denominazione_regione = np.array(new_df.denominazione_regione)
    warnings.filterwarnings("default")

    # Creation of the dataframes
    recovered_df = pd.DataFrame(recovered_data, columns = denominazione_regione, index = dates)
    new_daily_infections_df = pd.DataFrame(new_daily_infections_data, columns = denominazione_regione, index = dates)
    hospitalized_df = pd.DataFrame(hospitalized_data, columns = denominazione_regione, index = dates)
    deceased_df = pd.DataFrame(deceased_data, columns = denominazione_regione, index = dates)

    # fig, ax = plt.subplots(2, 2)
    # time_window = 60
    # ax[0,0].plot(recovered_df.iloc[-time_window:,:]) #the first row contains the names of the regions
    # ax[0,0].set_title("Recovered", fontsize = 10, y=1.0, pad=-14)
    # ax[0,1].plot(new_daily_infections_df.iloc[-time_window:,:])
    # ax[0,1].set_title("New_daily_infections", fontsize = 10, y=1.0, pad=-14)
    # ax[1,0].plot(hospitalized_df.iloc[-time_window:,:])
    # ax[1,0].set_title("Hospitalized", fontsize = 10, y=1.0, pad=-14)
    # ax[1,1].plot(deceased_df.iloc[-time_window:,:])
    # ax[1,1].set_title("Deceased", fontsize = 10, y=1.0, pad=-14)
    # fig.legend(denominazione_regione, loc = 'center right')
    # plt.subplots_adjust(left=0.06, bottom=0.1, right=0.8)
    # plt.show()

    return recovered_df, new_daily_infections_df, hospitalized_df, deceased_df


#This function creates a single dataframe containing the 4 variable obserbed in "region"
def create_dataframe_region(region):
    # Check if the most recent files have been downloaded
    Update_files.update()
    cwd = os.getcwd()
    regional_data_covid_path = os.path.join(cwd, "regional_data_covid")

    recovered_data = []
    new_daily_infections_data = []
    hospitalized_data = []
    deceased_data = []
    dates = []
    # Load data
    warnings.filterwarnings("ignore")
    for path in glob.glob(regional_data_covid_path):
        for i, subpath in enumerate(tqdm.tqdm(glob.glob(path + os.sep + "*"))):
            # add a new column to each dataframe

            new_df = pd.read_csv(subpath)  # , index_col=0
            dates.append(new_df.data[0])
            recovered_data.append(np.array(new_df.dimessi_guariti.iloc[region]))
            new_daily_infections_data.append(np.array(new_df.nuovi_positivi.iloc[region]))
            hospitalized_data.append(np.array(new_df.totale_ospedalizzati.iloc[region]))
            deceased_data.append(np.array(new_df.deceduti.iloc[region]))
    warnings.filterwarnings("default")

    # Creation of the dataframes
    recovered_df = pd.DataFrame(recovered_data, columns=["recovered"], index=dates)
    new_daily_infections_df = pd.DataFrame(new_daily_infections_data, columns=["new_daily_infections"], index=dates)
    hospitalized_df = pd.DataFrame(hospitalized_data, columns=["hospitalized"], index=dates)
    deceased_df = pd.DataFrame(deceased_data, columns=["deceased"], index=dates)

    return pd.concat([recovered_df, new_daily_infections_df, hospitalized_df, deceased_df], axis = 1)