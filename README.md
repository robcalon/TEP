# Transmission Expansion Planning
Package in progress that contains scripts to analyse performance of transmission networks in
the context of exploratory modelling and analysis.

## Work in progress
not all scripts are included yet, the full scripts are expected to be uploaded in the near future.

## Installation instructions
Conda-based installations has been tested on windows 10 and MacOS Catelina.
Perform all listed actions in conda (Anaconda Prompt) to use the repository.

#### Installation
Create new environment
````
$ conda create -n MyEnv python=3.7
````
Activate environment
````
$ conda activate MyEnv
````
Navigate to path of package
````
$ cd PATH/TO/transmission-expansion-planning
````
Install JupyterLab
````
$ conda install -c conda-forge jupyterlab nodejs geopandas descartes
````
(Optional) install ArcGIS. Only required if you want to fetch asset data stored on ArcGIS (Online)
````
$ conda install -c esri arcgis
````
Install all required packages
````
$ pip install -r requirements.txt
````
Install ipywidgets
````
jupyter labextension install @jupyter-widgets/jupyterlab-manager
````

#### Open JupyterLab

Activate environment
````
$ conda activate MyEnv
````
Navigate to path of package
````
$ cd ../transmission-expansion-planning
````
Launch Jupyter Lab
````
$ jupyter lab
````
