import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import datetime as dt
import warnings
import os
import glob
import csv
cwd = os.getcwd()
regional_data_covid_path = os.path.join(cwd,"regional_data_covid")

recovered_data = []
new_daily_infections_data = []
hospitalized_data = []
deceased_data = []
#Load data
warnings.filterwarnings("ignore")
for path in glob.glob(regional_data_covid_path):
    for i,subpath in enumerate(tqdm.tqdm(sorted(glob.glob(path + "/*")))):
        #add a new column to each dataframe

        new_df = pd.read_csv(subpath, index_col=0)
        recovered_data.append(np.array( new_df.dimessi_guariti))
        new_daily_infections_data.append(np.array(new_df.nuovi_positivi))
        hospitalized_data.append(np.array(new_df.totale_ospedalizzati))
        deceased_data.append(np.array(new_df.deceduti))
denominazione_regione = np.array(new_df.denominazione_regione)
warnings.filterwarnings("default")

# Creation of the dataframes
recovered_df = pd.DataFrame(recovered_data)
new_daily_infections_df = pd.DataFrame(new_daily_infections_data)
hospitalized_df = pd.DataFrame(hospitalized_data)
deceased_df = pd.DataFrame(deceased_data)

recovered_df.columns=denominazione_regione
new_daily_infections_df.columns=denominazione_regione
hospitalized_df.columns=denominazione_regione
deceased_df.columns=denominazione_regione

fig, ax = plt.subplots(2, 2)
time_window = 60
ax[0,0].plot(recovered_df.iloc[-time_window:,:]) #the first row contains the names of the regions
ax[0,0].set_title("Recovered", fontsize = 10, y=1.0, pad=-14)
ax[0,1].plot(new_daily_infections_df.iloc[-time_window:,:])
ax[0,1].set_title("New_daily_infections", fontsize = 10, y=1.0, pad=-14)
ax[1,0].plot(hospitalized_df.iloc[-time_window:,:])
ax[1,0].set_title("Hospitalized", fontsize = 10, y=1.0, pad=-14)
ax[1,1].plot(deceased_df.iloc[-time_window:,:])
ax[1,1].set_title("Deceased", fontsize = 10, y=1.0, pad=-14)
fig.legend(denominazione_regione, loc = 'center right')
plt.subplots_adjust(left=0.06, bottom=0.1, right=0.8)
plt.show()

abruzzo=[]
abruzzo.append(recovered_df["Abruzzo"])
abruzzo.append(hospitalized_df["Abruzzo"])
abruzzo.append(new_daily_infections_df["Abruzzo"])
abruzzo.append(deceased_df["Abruzzo"])

states=["Recovered", "Hospitalized", "Infected", "Deceased"];
abruzzo_df=pd.DataFrame(abruzzo).T;
abruzzo_df.columns=states;

exps_dir = "data_regional"
if not os.path.exists(exps_dir):
     os.makedirs(exps_dir)

 # save the result
savings_directory1=os.path.join(exps_dir, 'abruzzo.csv')
abruzzo_df.to_csv(savings_directory1, index=False)
#
# savings_directory2=os.path.join(exps_dir, 'infections.csv')
# new_daily_infections_df.to_csv(savings_directory2, index=False)
#
# savings_directory3=os.path.join(exps_dir, 'hospitalized.csv')
# hospitalized_df.to_csv(savings_directory3, index=False)
#
# savings_directory4=os.path.join(exps_dir, 'deceased.csv')
# deceased_df.to_csv(savings_directory4, index=False)
