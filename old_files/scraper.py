import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

def get_data(number_of_days=None, NoneChar = '-'):
    '''
    Accesses GitHub page and returns a list of composed of "number_of_days" components: each component is a pandas.DataFrame 
    object which contains the regional data, stored as in https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni
    (use https://github.com/pcm-dpc/COVID-19/blob/master/dati-regioni/dpc-covid19-ita-regioni-20220418.csv as reference).

    INPUTS:
    - number_of_days: e.g. if "number_of_days = 4" the function returns 4 csv files containing the 4 most recent csv files from GitHub repository.
    - NoneChar: char used to fill empty or missing data.
    '''

    print("=====================================================")
    print("Starting Scraping")

    dates = []
    table_list = []
    csv_names = []
    url = 'https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni'
    response = requests.get(url)
    if response.status_code != 200:
        print("Error: response_status != 200")
        return
    doc = BeautifulSoup(response.content, "html.parser")
    tags = doc.find_all("a")
    for tag in tags:
        n = str(tag.string)
        signature_string = "dpc-covid19-ita-regioni-2"
        if n[:len(signature_string)] == signature_string:
            dates.append(n)

    # start loop over days
    if number_of_days == None:
        selected_dates = dates
    else:
        selected_dates = dates[-number_of_days:]
    for count, additional_string in enumerate(selected_dates):
        new_url = url + "/" + additional_string
        response = requests.get(new_url)
        doc = BeautifulSoup(response.content, "html.parser")
        tbody = doc.tbody
        trs = tbody.contents
        trs_correct = []
        for t in trs:
            if t != '\n':
                trs_correct.append(t)
        # get the column names
        thead = doc.thead
        column_names = []
        for col_name in thead.find_all("th"):
            column_names.append(col_name.text)
        region_list = list(tbody.find_all("tr", class_="js-file-line"))
        region_list_clean = []
        for c in region_list:
            if c != '\n':
                region_list_clean.append(c)
        num_of_features = len(column_names) # == 24
        num_of_regions = len(region_list_clean) # == 21
        d = []
        numerical_features_index = [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21]
        for region_index in range(len(region_list_clean)):
            region_data = list(region_list_clean[region_index].find_all("td"))
            row = []
            for col in range(num_of_features):
                if region_data[col+1].string == None:
                    row.append(NoneChar)
                else:
                    row.append(region_data[col+1].string)
            d.append(row)
        # d is a list with 21 components, each of them is a list with 24 components

        # now i need to post process each component to transform string into numbers where necessary
        for region_index, region in enumerate(d):
            for feature_index in range(len(region)):
                if feature_index in numerical_features_index and region[feature_index] != NoneChar:
                    d[region_index][feature_index] = float(d[region_index][feature_index])

        # construct the dictionary and the pandas.DataFrame (organized as in the GitHub repo)
        data_dictionary = {}
        for i,cname in enumerate(column_names):
            l = []
            for region in d:
                l.append(region[i])
            data_dictionary[cname] = l
        table_list.append(pd.DataFrame(data=data_dictionary))
        csv_names.append(additional_string)
        print(additional_string,"|",count+1,"out of",len(selected_dates))

    print("=====================================================")
    print("Scraping Ended")

    return table_list

def get_csv_names(number_of_days=None, NoneChar = '-'):
    '''
    Returns the names of the csv files in the GitHub repository.
    Used to save the files into a dedicated folder.
    
    INPUTS:
    - number_of_days: e.g. if "number_of_days = 4" the function returns the names of the 4 most recent csv files from GitHub repository.
    - NoneChar: char used to fill empty or missing data.
    '''
    dates = []
    csv_names = []
    url = 'https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni'
    response = requests.get(url)
    if response.status_code != 200:
        print("Error: response_status != 200")
        return
    doc = BeautifulSoup(response.content, "html.parser")
    tags = doc.find_all("a")
    for tag in tags:
        n = str(tag.string)
        signature_string = "dpc-covid19-ita-regioni-2"
        if n[:len(signature_string)] == signature_string:
            dates.append(n)
    # start loop over days
    if number_of_days == None:
        selected_dates = dates
    else:
        selected_dates = dates[-number_of_days:]
    for additional_string in selected_dates:
        csv_names.append(additional_string)

    return csv_names

def save_data(table_list, directory_name, number_of_days=None, NoneChar = '-'):
    '''
    Saves the "table_list" of csv files into a directory which is named as in "directory_name".
    ADDITIONAL INPUTS:
    - number_of_days: used to call get_csv_names().
    - NoneChar: char used to fill empty or missing data.
    '''
    import os  
    os.makedirs(directory_name, exist_ok=True) 
    day_names = get_csv_names(number_of_days, NoneChar) 
    for index, day in enumerate(table_list):
        filename = directory_name + "/" + day_names[index]
        day.to_csv(filename, index = False)
    return 0

if __name__ == "__main__":
    foldername = "regional_data_covid"
    table_list = get_data()
    save_data(table_list, foldername)
    

