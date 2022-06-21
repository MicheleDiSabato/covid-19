import tqdm
import warnings
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import update_files

def create_dataframe():
    '''
    Function which returns the values of the 4 features for all 3 regions.
    '''
    # Check if the most recent files have been downloaded
    update_files.update()
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
        for i, subpath in enumerate(tqdm.tqdm(sorted(glob.glob(path + os.sep +"*")))):
            # add a new column to each dataframe
            new_df = pd.read_csv(subpath) 
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

    return recovered_df, new_daily_infections_df, hospitalized_df, deceased_df


#This function creates a single dataframe containing the 4 variable obserbed in "region"
def create_dataframe_region(region):
    '''
    Returns the data related to the resion identified by the index "region".
    Input: 
        region: (int) the index of the selected region (in alphabetical order: region != codice regione).
    Output: 
        two dataframe, according to which kind of infected individuals is needed:
        res1 uses "variazione_totale_positivi", res2 uses "nuovi_positivi"
    '''
    # Check if the most recent files have been downloaded
    update_files.update()
    cwd = os.getcwd()
    regional_data_covid_path = os.path.join(cwd, "regional_data_covid")

    recovered_data = []
    nuovi_positivi = []
    variazione_totale_positivi = []
    hospitalized_data = []
    deceased_data = []
    dates = []
    # Load data
    warnings.filterwarnings("ignore")
    for path in glob.glob(regional_data_covid_path):
        for i, subpath in enumerate(tqdm.tqdm(sorted(glob.glob(path + os.sep + "*")))):
            # add a new column to each dataframe
            new_df = pd.read_csv(subpath) 
            dates.append(new_df.data[0])
            recovered_data.append(np.array(new_df.dimessi_guariti.iloc[region]))
            nuovi_positivi.append(np.array(new_df.nuovi_positivi.iloc[region]))
            variazione_totale_positivi.append(np.array(new_df.variazione_totale_positivi.iloc[region]))
            hospitalized_data.append(np.array(new_df.totale_ospedalizzati.iloc[region]))
            deceased_data.append(np.array(new_df.deceduti.iloc[region]))
    warnings.filterwarnings("default")

    # Create the dataframes
    recovered_df = pd.DataFrame(recovered_data, columns=["recovered"], index=dates)
    new_positives_df = pd.DataFrame(nuovi_positivi, columns=["new_daily_infections"], index=dates)
    variation_total_positives_df = pd.DataFrame(variazione_totale_positivi, columns=["new_daily_infections"], index=dates)
    hospitalized_df = pd.DataFrame(hospitalized_data, columns=["hospitalized"], index=dates)
    deceased_df = pd.DataFrame(deceased_data, columns=["deceased"], index=dates)
    res1 = pd.concat([recovered_df, hospitalized_df, deceased_df, variation_total_positives_df], axis = 1)
    res2 = pd.concat([recovered_df, hospitalized_df, deceased_df, new_positives_df], axis = 1)

    # res1 uses "variazione_totale_positivi", res2 uses "nuovi_positivi"
    return res1, res2
    
if __name__ == "__main__":
    '''
    Create 3 csv files, each for every region.
    '''
    # Lombardia: 8
    # Lazio: 6
    # Sicilia: 16
    d1_sicilia, d2_sicilia = create_dataframe_region(16)
    d_sicilia = d2_sicilia
    d1_lombardia, d2_lombardia = create_dataframe_region(8)
    d_lombardia = d2_lombardia
    d1_lazio, d2_lazio = create_dataframe_region(6)
    d_lazio = d2_lazio
    for feature in range(d_lombardia.shape[1]):
        with open("lombardia_csv/"+"lombardia_"+d_lombardia.columns.values[feature] + ".csv", 'w') as f:
            for day in range(d_lombardia.shape[0]):
                f.write(str(d_lombardia.iloc[day, feature]) + '\n')
        with open("lazio_csv/"+"lazio_"+d_lazio.columns.values[feature] + ".csv", 'w') as f:
            for day in range(d_lazio.shape[0]):
                f.write(str(d_lazio.iloc[day, feature]) + '\n')
        with open("sicilia_csv/"+"sicilia_"+d_sicilia.columns.values[feature] + ".csv", 'w') as f:
            for day in range(d_sicilia.shape[0]):
                f.write(str(d_sicilia.iloc[day, feature]) + '\n')
    with open("lombardia_csv/"+"lombardia_all.csv", "w") as f:
        for day in range(d_lombardia.shape[0]):
            f.write(str(d_lombardia.iloc[day, 0]) + "," + str(d_lombardia.iloc[day, 1]) + "," + str(d_lombardia.iloc[day, 2]) + "," + str(d_lombardia.iloc[day, 3]) + '\n')

