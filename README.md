# DataStreams

This project explores machine learning methods for data streams, focusing on the regression task of predicting travel time and delay across different transportation modes, such as taxis and airplanes. 

The repository is organized as follows:

- `dataset/` - real and synthetic datastes, scripts for their generation or preprocessing
- `transformer/` - transformation of the datasets .csv files into stream, config files
- `ensemble/` – implementation of adaptive regression ensemble
- `centroid_drift/` – implementation of centroid drift detector
- `results/` – computed results
- `results_drift/` – computed results for drift detection and visualization for drift experiments
- `plots/` – visualizations of experimental results
- `initial_experiments/` – initial scripts with experiments
- `nybb_16b/` – folder with geojson files for division into districts of Taxi dataset
  
  

Real datasets:
- [Taxi dataset (2016 NYC Yellow Cab trip record data)](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)
- [Airplane dataset (Data Expo 2009: Airline on time data)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7)
