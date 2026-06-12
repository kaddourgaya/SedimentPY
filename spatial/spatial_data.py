import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from pyproj import Transformer


class SpatialData:
    """
    Python equivalent of CEQUEAUSpatialData.m
    Loads:
        - DEM, FAC, slope, flowpathlength
        - CE_fishnet.shp
        - CP_fishnet.shp
        - streams_cequeau.shp
        - rtable.csv
    """

    def __init__(self, physiography_folder, result_folder):
        print("LOADING CEQUEAU SPATIAL DATA...")

        # ---------------------------------------------------------
        # 1. DEM + reference + lat/lon detection
        # ---------------------------------------------------------
        dem_path = f"{physiography_folder}/DEM.tif"
        with rasterio.open(dem_path) as src:
            self.DEM = src.read(1).astype(float)
            self.R_DEM = src.transform
            crs = src.crs
            bounds = src.bounds

        # Detect lat/lon center
        if crs.is_geographic:
            # Already lat/lon
            self.longitude = (bounds.left + bounds.right) / 2
            self.latitude = (bounds.top + bounds.bottom) / 2
        else:
            # Projected → convert to lat/lon
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
            self.longitude = (lon_min + lon_max) / 2
            self.latitude = (lat_min + lat_max) / 2

        # ---------------------------------------------------------
        # 2. Other rasters
        # ---------------------------------------------------------
        self.FAC = self._read_raster(f"{physiography_folder}/FAC.tif")
        self.slope_meter = self._read_raster(f"{physiography_folder}/Slope.tif")
        self.flowpathlength = self._read_raster(f"{physiography_folder}/FlowPathLength.tif")

        # CE grid reference
        with rasterio.open(f"{physiography_folder}/CEgrid.tif") as src:
            self.R_CE = src.transform

        # ---------------------------------------------------------
        # 3. Shapefiles
        # ---------------------------------------------------------
        self.CEfishnet = gpd.read_file(f"{physiography_folder}/CE_fishnet.shp")
        self.CPfishnet = gpd.read_file(f"{physiography_folder}/CP_fishnet.shp")
        self.stream_cequeau = gpd.read_file(f"{physiography_folder}/streams_cequeau.shp")

        # ---------------------------------------------------------
        # 4. Routing table
        # ---------------------------------------------------------
        self.rtable = pd.read_csv(f"{result_folder}/rtable.csv")

        # User-defined outlet (same as MATLAB)
        self.Outlet = 61

        print("SPATIAL DATA LOADED.")

    # -------------------------------------------------------------
    # Helper: read raster as float
    # -------------------------------------------------------------
    def _read_raster(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(float)
