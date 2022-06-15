import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import july
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import glob
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
tfkl = tf.keras.layers

def from_region_to_index(region_name=None, codice_regione = False):
    '''
    if codice_regione:
        outputs the "codice_regione" associated to the region called "region_name". 
    else:
        outputs the alphabetical index associated to the region called "region_name".
    '''
    d = {}
    cwd = os.getcwd()
    first_csv_path = os.path.join(cwd, "regional_data_covid" + os.sep + 'dpc-covid19-ita-regioni-20200224.csv')
    df = pd.read_csv(first_csv_path) 
    if codice_regione:
        for i in range(df.shape[0]):
            d[df.iloc[i]['denominazione_regione']] = int(df.iloc[i]['codice_regione'])
        if region_name==None:
            res = d
        else:
            res = d[region_name]
        return res
    else:
        for i in range(df.shape[0]):
            d[df.iloc[i]['denominazione_regione']] = i
        if region_name == None:
            return -1
        else:
            return d[region_name] 
    

def inspect_dataframe(df, columns = None):
    '''
    Plots the columns of the datafram df.
    '''
    if columns == None:
        columns = df.columns
    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(df[col].values)
        axs[i].set_title(col)
    plt.show()
    
def inspect_multivariate(X, y, columns, telescope, idx=None, plot_type = "line"):
    '''
    Inspect one random minibatch (X is one minibatch obtained as a result of build_sequences).
    '''
    if(idx==None):
        idx=np.random.randint(0,len(X))
    
    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(np.arange(len(X[0,:,i])), X[idx,:,i])
        if plot_type == "points":
            axs[i].scatter(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
        if plot_type == "line":
            axs[i].plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
        axs[i].set_title(col)
        # axs[i].set_ylim(0-0.1,1+0.1)
    plt.show()
    
def inspect_univariate(X, y, columns, telescope, idx=None, plot_type = "line"):
    '''
    Inspect one random minibatch (X is one minibatch obtained as a result of build_sequences).
    '''
    if(idx==None):
        idx=np.random.randint(0,len(X))
    plt.figure(figsize=(10, 5))
    for i, col in enumerate(columns):
        plt.plot(np.arange(len(X[0,:,i])), X[idx,:,i])
        if plot_type == "points":
            plt.scatter(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
        if plot_type == "line":
            plt.plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
    plt.title(columns[0])
    plt.show()

def inspect_multivariate_prediction(X, y, pred, columns, telescope, idx=None, plot_type = "line"):
    '''
    [multivariate version]
    Plots the batch X, the true target y, the prediction pred.
    '''
    if(idx==None):
        idx=np.random.randint(0,len(X))

    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(np.arange(len(X[0,:,i])), X[idx,:,i])
        if plot_type == "line":
            axs[i].plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
            axs[i].plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), pred[idx,:,i], color='green') 
        if plot_type == "points":
            axs[i].scatter(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
            axs[i].scatter(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), pred[idx,:,i], color='green')             
        axs[i].set_title(col)
        # axs[i].set_ylim(0-0.1,1+0.1)
    plt.show()

def inspect_univariate_prediction(X, y, pred, columns, telescope, idx=None, plot_type = "line"):
    '''
    [univariate version]
    Plots the batch X, the true target y, the prediction pred.
    '''
    if(idx==None):
        idx=np.random.randint(0,len(X))
    plt.figure(figsize = (10,5))
    for i, col in enumerate(columns):
        plt.plot(np.arange(len(X[0,:,i])), X[idx,:,i])
        if plot_type == "line":
            plt.plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
            plt.plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), pred[idx,:,i], color='green') 
        if plot_type == "points":
            plt.scatter(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:], color='orange')
            plt.scatter(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), pred[idx,:], color='green')             
    plt.title(columns[0])
    plt.show()
    
def build_sequences(df, target_labels, days, window=100, stride=20, telescope=7):
    '''
    Constuct training/test batch sequences.
    '''
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    days_strings = []
    temp_df = df.copy().values
    temp_label = df[target_labels].copy().values
    padding_len = len(df)%window

    if(padding_len != 0):
        # Compute padding length
        padding_len = window - len(df)%window
        padding = np.zeros((padding_len,temp_df.shape[1]), dtype='float64')
        temp_df = np.concatenate((padding,df))
        padding = np.zeros((padding_len,temp_label.shape[1]), dtype='float64')
        temp_label = np.concatenate((padding,temp_label))
        days.insert(0, "-")
        assert len(temp_df) % window == 0

    for idx in np.arange(0,len(temp_df)-window-telescope,stride):
        dataset.append(temp_df[idx:idx+window])
        labels.append(temp_label[idx+window:idx+window+telescope])
        days_strings.append(days[idx+window])
            
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels, days_strings

def from_date_to_index(date):
    '''
    Returns the index corresponding to the selected date.
    "date" needs to be in a format such as: "20200224".
    Returns -1 if "date" is not included (for debugging).
    '''
    cwd = os.getcwd()
    regional_data_covid_path = os.path.join(cwd, "regional_data_covid")    
    warnings.filterwarnings("ignore")
    index = -1
    date_names = []
    for path in glob.glob(regional_data_covid_path):
        for i, subpath in enumerate(sorted(glob.glob(path + os.sep + "*"))):
            s = subpath.split(os.sep)[-1]
            s = s.split(".")[0]
            s = s.split("-")[-1]
            if s == date:
                index = i
            date_names.append(s)
    return (date_names, index)

def parse_dates(dates_string):
    '''
    Modifies data format for calendar plots.
    Input "dates_string" is a list of strings.
    '''
    res = []
    for date in dates_string:
        d = date[:4] + "-" + date[4:6] + "-" + date[6:]
        res.append(d)
    return res
    
    
def from_date_to_sample(date, data_np, window = 28, telescope = 7, delta_vis = 50, verbose = 0):
    '''
    From "date" produces one sample/batch on which to test the model.
    Example: 
    d, y = from_date_to_sample("20211231", X_train_raw.values, verbose = 3, window = 28, telescope = 7, delta_vis = 100)
    '''
    (date_names, date_index) = from_date_to_index(date) # starts from 0, è la data a partire dalla quale voglio fare la previsione
    pd = parse_dates([date])
    assert date_index != -1, "Date " + pd[0] + " inserted is not valid"
    assert date_index - window >= 0, pd[0] + ": select a more recent date"
    assert date_index + telescope <= data_np.shape[0], pd[0] + ": select a less recent date"
    d = np.zeros((1, window, data_np.shape[1]))
    y = np.zeros((1, telescope, data_np.shape[1]))
    d[0] = data_np[date_index-window:date_index, :]
    y[0] = data_np[date_index:date_index+telescope, :]
    
    date_names_interval = parse_dates(date_names[date_index-window:date_index+telescope])
    date_names_total = parse_dates(date_names)
    if verbose >= 1:
        print("time interval selected: ", date_names_interval)
    if verbose >= 2:
        left = date_index - delta_vis
        right = date_index + delta_vis
        if left < 0:
            left = 0
        if right > len(date_names):
            right = len(date_names)
        from july.utils import date_range
        dates = date_range(date_names_total[left], date_names_total[right])
        color_intensity = np.zeros((len(date_names),))
        for i in range(date_index-window, date_index+telescope+1, 1):
            color_intensity[i] = telescope + window - abs(i - date_index)
        color_intensity[date_index] = 1.5*color_intensity[date_index]
        color_intensity = color_intensity[left:right]
        july.heatmap(dates, color_intensity, title='Dates Selected', cmap="golden", horizontal = True, date_label=True)
    return (d, y)

def test_on_samples(model, dates_chosen, model_info, verbose = 0, prediction_type = "autoregressive"):
    '''
    Reads the dates contained in the list dates_chosen, loads the model and the corresponding information 
    (contained in the dictionary model_info), calls build_sequences to construct the samples on which to test the model
    and finlly evaluates the model on the samples.
    Returns the data for the test samples, the true target, the prediction and the L2 errors.
    E.g.
    dates_chosen = ["20201231", "20211231", "20200801", "20210801", "20200401", "20210401"]
    model_info = {'window': 31, 'telescope': 1, 'reg_telescope': 7, 'n_features': 1, 'complete_dataset': X_train_raw}
    '''
    X_train_raw = model_info['complete_dataset']
    window = model_info['window']
    telescope = model_info['telescope']
    reg_telescope = model_info['reg_telescope']
    p = model_info['n_features']
    X_test_samples = np.zeros((len(dates_chosen), window, p))
    if prediction_type == "oneshot":
        y_test_samples = np.zeros((len(dates_chosen), telescope, p))
        preds = np.zeros((len(dates_chosen), telescope, p))
    if prediction_type == "autoregressive":
        y_test_samples = np.zeros((len(dates_chosen), reg_telescope, p)) 
        preds = np.zeros((len(dates_chosen), reg_telescope, p))
    for i, date in enumerate(dates_chosen):
        (X_test_samples[i], y_test_samples[i]) = from_date_to_sample(date, X_train_raw.values, window = window, verbose = verbose)
    errors = np.zeros((p,1))
    for i in range(X_test_samples.shape[0]):
        sample = X_test_samples[i]
        X_temp = np.zeros((1, window, p))
        X_temp[0] = sample
        if prediction_type == "autoregressive":
            # Autoregressive Forecasting
            predictions = np.array([])
            for reg in range(0,reg_telescope,telescope):
                pred_temp = model.predict(X_temp)
                if len(predictions)==0:
                    predictions = pred_temp
                else:
                    predictions = np.concatenate((predictions,pred_temp),axis=1)
                if len(pred_temp.shape) == 2:
                    t = np.zeros((1, pred_temp.shape[0], pred_temp.shape[1]))
                    t[0] = pred_temp
                    pred_temp = t
                X_temp = np.concatenate((X_temp[:,telescope:,:],pred_temp), axis=1)
        if prediction_type == "oneshot":
            predictions = model.predict(X_temp)
        if len(predictions.shape) == 2:
            t = np.zeros((1, predictions.shape[0], predictions.shape[1]))
            t[0] = predictions
            predictions = t
        print("Processing date", dates_chosen[i])
        preds[i] = predictions[0,:,:].reshape(-1,1)
        for j in range(p):
            temp = np.sum((y_test_samples[i, :, j].reshape(-1,1) - predictions[0, :, j].reshape(-1,1))**2)
            errors[j, 0] += np.sqrt(temp)
        
    errors = errors / X_test_samples.shape[0]
    return (X_test_samples, y_test_samples, preds, errors)
