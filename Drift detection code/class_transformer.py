from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from river import drift
from river import base
from river.base import DriftDetector
from river.stats._rust_stats import AdaptiveWindowing

us_regions = {
    'South': [
        'TX', 'VA', 'MS', 'FL', 'AR', 'LA', 'OK', 'NC', 
        'KY', 'AL', 'TN', 'MD', 'SC', 'WV', 'GA'
    ],
    'West': [
        'NV', 'CA', 'OR', 'AZ', 'WA', 'UT', 'NM', 'ID', 
        'CO', 'WY', 'HI', 'MT', 'AK'
    ],
    'Midwest': [
        'IN', 'MO', 'IL', 'NE', 'OH', 'MI', 'WI', 'KS', 
        'IA', 'MN', 'SD', 'ND'
    ],
    'Northeast': [
        'NY', 'NH', 'PA', 'RI', 'CT', 'NJ', 'VT', 'ME', 'MA'
    ],
    'Other': [
        'PR', 'VI' 
    ]
}

class FeatureDistrict(base.Transformer):
 
    def __init__(self, dataset=None, columns_to_drop=None):
        self.dataset = dataset
        self.columns_to_drop = columns_to_drop or []
        self.us_regions = {
            'South':     ['TX', 'VA', 'MS', 'FL', 'AR', 'LA', 'OK', 'NC',
                          'KY', 'AL', 'TN', 'MD', 'SC', 'WV', 'GA'],
            'West':      ['NV', 'CA', 'OR', 'AZ', 'WA', 'UT', 'NM', 'ID',
                          'CO', 'WY', 'HI', 'MT', 'AK'],
            'Midwest':   ['IN', 'MO', 'IL', 'NE', 'OH', 'MI', 'WI', 'KS',
                          'IA', 'MN', 'SD', 'ND'],
            'Northeast': ['NY', 'NH', 'PA', 'RI', 'CT', 'NJ', 'VT', 'ME', 'MA'],
            'Other':     ['PR', 'VI'],
        }
 

        if self.dataset == 'Taxi':
            nyc_boroughs = gpd.read_file("nybb.shp").to_crs(epsg=4326)
            self.boroughs_list = [(row['geometry'], row['BoroName'])
                for i, row in nyc_boroughs.iterrows()
            ]
            self.boroughs_list = [(i, j) for i, j in self.boroughs_list if j != "Staten Island" and j!="Bronx"
]
        else:
            self.boroughs_list = []
 
    def get_region_taxi(self, lat, lon):
        punkt = Point(lon, lat)
        for b_geom, b_name in self.boroughs_list:
            if b_geom.contains(punkt):
                return b_name
        return "Other"
 
    def get_region_airplanes(self, state_code):
        for region, states in self.us_regions.items():
            if state_code in states:
                return region
        return "Other"
 
    def timestamp_airplanes(self, x):
        month = int(x["Month"])
        day   = int(x["DayofMonth"])
        h     = int(x["CRSDepTime"] // 100)
        x["timestamp"] = datetime(2008, month, day, h)
        return x
 
    def timestamp_taxi(self, x):
        dt = x["pickup_datetime"]
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        x["timestamp"] = datetime(dt.year, dt.month, dt.day, dt.hour)
        return x
    
    def timestamp_warsaw(self, x):
        dt = x["timestamp"]
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        x["timestamp"] = datetime(dt.year, dt.month, dt.day, dt.hour)
        return x
        
 
    def transform_one(self, x):
        x = x.copy()
 
        if self.dataset=='Taxi':
            x = self.timestamp_taxi(x)
            pickup_d  = self.get_region_taxi(x["pickup_latitude"],  x["pickup_longitude"])
            dropoff_d = self.get_region_taxi(x["dropoff_latitude"], x["dropoff_longitude"])
 
        elif self.dataset=='Airplanes':
            x = self.timestamp_airplanes(x)
            pickup_d  = self.get_region_airplanes(x["origin_state"])
            dropoff_d = self.get_region_airplanes(x["dest_state"])
        elif self.dataset=='Warsaw':
            x = self.timestamp_warsaw(x)
            pickup_d=x["pickup_d"]
            dropoff_d= x["dropoff_d"]
 
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}.")
        x["pickup_district"]  = pickup_d
        x["dropoff_district"] = dropoff_d
        if self.dataset !="Warsaw":
            x["is_weekend"] = int(x["timestamp"].weekday() >= 5)
        x["within_district"]  = int(
            pickup_d == dropoff_d and pickup_d not in ("Other", "Outer")
        )
 
        for col in self.columns_to_drop:
            x.pop(col, None)
 
        return x

        
