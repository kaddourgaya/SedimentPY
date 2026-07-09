import rasterio
import numpy as np
import json
import os
from pyproj import Transformer

class SpatialData:
    def __init__(self, physiography_folder, result_folder):
        print("LOADING CEQUEAU SPATIAL DATA...")

        self.physiography_folder = physiography_folder
        self.result_folder = result_folder

        # DEM
        dem_path = f"{physiography_folder}/DEM.tif"
        with rasterio.open(dem_path) as src:
            self.DEM = src.read(1).astype(float)
            self.R_DEM = src.transform
            crs = src.crs
            bounds = src.bounds

        # Lat/lon
        if crs.is_geographic:
            self.longitude = (bounds.left + bounds.right) / 2
            self.latitude  = (bounds.top + bounds.bottom) / 2
        else:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
            self.longitude = (lon_min + lon_max) / 2
            self.latitude  = (lat_min + lat_max) / 2

        # Other rasters
        self.FAC = self._read_raster(f"{physiography_folder}/FAC.tif")
        self.slope_meter = self._read_raster(f"{physiography_folder}/Slope.tif")
        self.flowpathlength = self._read_raster(f"{physiography_folder}/FlowPathLength.tif")

        # Load CE/CP from bassinVersant.json
        json_path = os.path.join(result_folder, "bassinVersant.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"bassinVersant.json introuvable : {json_path}")

        with open(json_path, "r") as f:
            bv = json.load(f)

        self.CE = bv["carreauxEntiers"]
        self.CP = bv["carreauxPartiels"]

        print("SPATIAL DATA LOADED.")

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(float)
