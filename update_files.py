import scrape
import os
import glob
from datetime import datetime
from datetime import date


folder_name = "regional_data_covid"
beginning = date(2020,2,24)

def check_new_csv():
    '''
    Returns the number of days that need to be downloaded from the repository.
    '''
    count = 0 
    for path in glob.glob(folder_name + os.sep + "*"):
        if path.split(os.sep)[-1].startswith("dpc"):
            count = count + 1
    return int((date.today() - beginning).days - count)


def update():
    '''
    If the local folder is not up-to-date, this functions downloads the missing new data from the repository.
    '''
    number_of_days = check_new_csv()
    if(number_of_days>0):
        scrape.scrape(folder_name, number_of_days = number_of_days)
    else:
        print("The folder is already up to date")
