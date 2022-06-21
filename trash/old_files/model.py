#install scalecast package

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
sns.set(rc={'figure.figsize':(15,8)})
import create_dataframes
import statsmodels.api as sm


recovered_df, new_daily_infections_df, hospitalized_df, deceased_df = create_dataframes.create_dataframe()
index = 3 #Lombardy index
dataframe = create_dataframes.create_dataframe_region(index)
print(dataframe) # dataframe lombardy
print(recovered_df)
#lets try to just focus on lombardy's new daily infections
#dates = new_daily_infections_df.index
#print(new_daily_infections_df.index)
#f = Forecaster(y=new_daily_infections_df.iloc[:,index],
#               current_dates=dates)

##partial auto correlation plot
#f.plot_acf()
#f.plot_pacf()
#plt.show()
#plt.show()

##decomposition of its trend, seasonal, and residual parts
#f.seasonal_decompose(diffy=True, period = 1).plot()
#plt.show()