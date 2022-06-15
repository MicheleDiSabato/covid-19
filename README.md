# Authors: 
* [Federica Botta](https://www.linkedin.com/in/federica-botta-8629391b3/) 
* [Simone Colombara](https://www.linkedin.com/in/simone-colombara-a4a430167/)
* [Michele Di Sabato](https://www.linkedin.com/in/michele-di-sabato/)

# Covid 19:
In this project we use RNNs to predict the weekly values of four features related to the COVID 19 pandemic in Sicily, Lombardy and Lazio (IT).

Each function is explained in the [Code](#Code) section.

## Dataset:
The source of the dataset is the [GitHub repository](https://github.com/pcm-dpc/COVID-19) of the [Protezione Civile](https://en.wikipedia.org/wiki/Protezione_Civile), specifically [this link](https://github.com/pcm-dpc/COVID-19/tree/master/dati-province). We focused on the following features:
1. <em>nuovi_positivi</em>: **daily** amount of current positive cases (totale_casi current day - totale_casi previous day)
2. <em>totale_ospedalizzati</em>: total **cumulative** hospitalised patients
3. <em>dimessi_guariti</em>:**daily** amount of recovered
4. <em>deceduti</em>: total **cumulative** number of deceased

We only focus on:
1. Sicily
2. Lombardy
3. Lazio

## Goal:
The goal is twofold:
1. set up an **automatic tool** that accesses, updates and organizes the COVID-19 epidemiological data of Italy (on a regional basis) as explained in the [Dataset](#dataset) section
2. **predict** four variables regarding the COVID19 pandemic situation in three regions in Italy

## Scraping:
To download automatically the data from the [repository](https://github.com/pcm-dpc/COVID-19) we used the libraries [`requests`](https://pypi.org/project/requests/) and [`bs4`](https://pypi.org/project/beautifulsoup4/). To minimize scraping time, the function [`update_files.csv`]() checks for new additions to the repository, to avoid dowloading files which are already present in the local folder. It's best to periodically download the entire folder from scratch, since sometimes the maintainers of the repository will change past csv files to correct mistakes or wrongly reported data.

## Data preprocessing:

## Model:

## Predictions:

## Conclusions:

## Code and structure of the repository: 
```
COVID-19/
│
├── aree/
│   ├── geojson
│   │   ├── dpc-covid-19-ita-aree-comuni.geojson
│   │   ├── dpc-covid19-ita-aree.geojson
│   ├── shp
│   │   ├── dpc-covid19-ita-aree-comuni.dbf
│   │   ├── dpc-covid19-ita-aree-comuni.prj
│   │   ├── dpc-covid19-ita-aree-comuni.shp
│   │   ├── dpc-covid19-ita-aree-comuni.shx
│   │   ├── dpc-covid19-ita-aree.dbf
│   │   ├── dpc-covid19-ita-aree.prj
│   │   ├── dpc-covid19-ita-aree.shp
│   │   ├── dpc-covid19-ita-aree.shx
├── dati-andamento-nazionale/
│   ├── dpc-covid19-ita-andamento-nazionale-*.csv
│   ├── dpc-covid19-ita-andamento-nazionale-latest.csv
│   ├── dpc-covid19-ita-andamento-nazionale.csv
├── dati-contratti-dpc-forniture/
│   ├── dpc-covid19-dati-contratti-dpc-forniture.csv
│   ├── dpc-covid19-dati-pagamenti-contratti-dpc-forniture.csv
│   ├── dati-json
│   │   ├── dpc-covid19-dati-contratti-dpc-forniture.csv
│   │   ├── dpc-covid19-dati-pagamenti-contratti-dpc-forniture.csv
│   ├── file-atti-negoziali
│   │   ├── dpc-contratto-covid19-*.pdf
├── dati-json/
│   ├── dpc-covid19-ita-andamento-nazionale-latest.json
│   ├── dpc-covid19-ita-andamento-nazionale.json
│   ├── dpc-covid19-ita-note-en.json
│   ├── dpc-covid19-ita-note-it.json
│   ├── dpc-covid19-ita-province-latest.json
│   ├── dpc-covid19-ita-province.json
│   ├── dpc-covid19-ita-regioni-latest.json
│   ├── dpc-covid19-ita-regioni.json
├── dati-province/
│   ├── dpc-covid19-ita-province-*.csv
│   ├── dpc-covid19-ita-province-latest.csv
│   ├── dpc-covid19-ita-province.csv
├── dati-regioni/
│   ├── dpc-covid19-ita-regioni-*.csv
│   ├── dpc-covid19-ita-regioni-latest.csv
│   ├── dpc-covid19-ita-regioni.csv
├── metriche
│   ├── dpc-covid19-ita-metriche-dashboard-desktop.csv
│   ├── dpc-covid19-ita-metriche-dashboard-desktop.json
│   ├── dpc-covid19-ita-metriche-dashboard-mobile.csv
│   ├── dpc-covid19-ita-metriche-dashboard-mobile.json
├── note/
│   ├── dpc-covid19-ita-note-en.csv
│   ├── dpc-covid19-ita-note-it.csv
├── schede-riepilogative/
│   ├── province
│   │   ├── dpc-covid19-ita-scheda-province-*.pdf
│   ├── regioni
│   │   ├── dpc-covid19-ita-scheda-regioni-*.pdf
```





















