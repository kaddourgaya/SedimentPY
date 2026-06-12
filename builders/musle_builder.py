import numpy as np
import rasterio
import os
from math import exp
from datetime import datetime


class MUSLEBuilder:
    """
    Python equivalent of MUSLEBuilder.m
    Computes:
        - K factor
        - LS factor
        - P factor
        - CFRG
        - NDVI baseline
        - NDVI temporal
        - C_static
    """

    def __init__(self, geo_dir, soil_dir, ndvi_dir, lulc_dir, output_dir, spatial):
        self.geo_dir = geo_dir
        self.soil_dir = soil_dir
        self.ndvi_dir = ndvi_dir
        self.lulc_dir = lulc_dir
        self.output_dir = output_dir
        self.spatial = spatial

        os.makedirs(output_dir, exist_ok=True)

        print("\nBuilding MUSLE structural rasters...")

        # Storage
        self.K = None
        self.LS = None
        self.P = None
        self.CFRG_raw = None
        self.NDVI_baseline = None
        self.NDVI_temporal = None
        self.NDVI_curve_annual = None
        self.ndvi_min = None
        self.ndvi_max = None

    # -------------------------------------------------------------
    # Helper: read raster
    # -------------------------------------------------------------
    def _read(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(float), src.transform

    # -------------------------------------------------------------
    # K FACTOR
    # -------------------------------------------------------------
    def compute_K_factor(self, sand, silt, clay, orgC):
        TSAND, R = self._read(f"{self.soil_dir}/{sand}")
        TSILT, _ = self._read(f"{self.soil_dir}/{silt}")
        TCLAY, _ = self._read(f"{self.soil_dir}/{clay}")
        ORGC, _ = self._read(f"{self.soil_dir}/{orgC}")

        fcsand = 0.2 + 0.3 * np.exp(-0.256 * TSAND * (1 - TSILT / 100))
        fcl_si = (TSILT / (TCLAY + TSILT)) ** 0.3
        forg = 1 - ((0.25 * ORGC) / (ORGC + np.exp(3.72 - 2.95 * ORGC)))
        denom = (1 - TSAND / 100) + np.exp(-5.51 + 22.9 * (1 - TSAND / 100))
        fhisand = 1 - (0.7 * (1 - TSAND / 100) / denom)

        self.K = fcsand * fcl_si * forg * fhisand * 0.1317

        print("✓ K-factor computed")

    # -------------------------------------------------------------
    # LS FACTOR
    # -------------------------------------------------------------
    def compute_LS_factor(self, slope_file, fac_file, dem_file, method="desmet_govers"):
        slope_deg, R = self._read(f"{self.geo_dir}/{slope_file}")
        flow_acc, _ = self._read(f"{self.geo_dir}/{fac_file}")
        dem, _ = self._read(f"{self.geo_dir}/{dem_file}")

        slope_rad = np.deg2rad(slope_deg)
        cell_size = R.a  # pixel size

        if method == "desmet_govers":
            F = np.sin(slope_rad) / (0.0896 * (np.sin(slope_rad) ** 0.8 + 0.56))
            m = F / (F + 1)
            A = flow_acc * cell_size**2

            L = ((A + cell_size**2) ** (m + 1) - A ** (m + 1)) / (
                cell_size ** (m + 2) * 22.13**m
            )

            S = np.zeros_like(slope_rad)
            mask1 = slope_rad < 0.09
            mask2 = ~mask1
            S[mask1] = 10.8 * np.sin(slope_rad[mask1]) + 0.03
            S[mask2] = 16.8 * np.sin(slope_rad[mask2]) - 0.50

            LS = L * S

            thr = np.maximum(100, 500 * np.sin(slope_rad))
            LS[LS > thr] = np.nan

            self.LS = LS

        else:
            raise NotImplementedError(f"LS method {method} not implemented.")

        print("✓ LS-factor computed")

    # -------------------------------------------------------------
    # P FACTOR
    # -------------------------------------------------------------
    def compute_P_factor(self, lulc_file, slope_file, isPEI=False):
        LULC, R = self._read(f"{self.lulc_dir}/{lulc_file}")
        slope, _ = self._read(f"{self.geo_dir}/{slope_file}")

        P_static = np.full_like(LULC, np.nan)

        agCodes = [133,136,145,146,147,153,155,158,162,177,179,182,195]
        agMask = np.isin(LULC, agCodes)

        P_pei = np.array([[0,3,0.50],[3,8,0.38],[8,15,0.25]])
        P_canada = np.array([[0,5,0.10],[5,15,0.12],[15,30,0.14],[30,50,0.19],[50,75,0.25],[75,100,0.33]])

        rows, cols = np.where(agMask)

        for r, c in zip(rows, cols):
            s = slope[r, c]
            table = P_pei if isPEI else P_canada
            idx = np.where((s >= table[:,0]) & (s < table[:,1]))[0]
            if len(idx) > 0:
                P_static[r, c] = table[idx[0], 2]

        P_static[np.isin(LULC, [20,80])] = np.nan
        P_static[np.isin(LULC, [30,34])] = 1.0
        P_static[np.isin(LULC, [50,110,122,131,142])] = 1.0
        P_static[np.isin(LULC, [210,220,230])] = 1.0

        self.P = P_static

        print("✓ P-factor computed")

    # -------------------------------------------------------------
    # CFRG
    # -------------------------------------------------------------
    def compute_CFRG_raw(self, cfrg_file):
        coarse, R = self._read(f"{self.soil_dir}/{cfrg_file}")
        self.CFRG_raw = coarse.astype(float)
        print("✓ CFRG raw loaded")

    # -------------------------------------------------------------
    # LOADERS (MATLAB-style)
    # -------------------------------------------------------------
    def load_K_factor(self):
        self.K, _ = self._read("data/MUSLE/K_factor.tif")
        print("✓ K-factor loaded from MATLAB raster")

    def load_LS_factor(self):
        self.LS, _ = self._read("data/MUSLE/LS_factor.tif")
        print("✓ LS-factor loaded from MATLAB raster")

    def load_P_factor(self):
        self.P, _ = self._read("data/MUSLE/P_factor.tif")
        print("✓ P-factor loaded from MATLAB raster")

    def load_CFRG_factor(self):
        self.CFRG_raw, _ = self._read("data/MUSLE/CFRG_factor.tif")
        print("✓ CFRG-factor loaded from MATLAB raster")
    
    # -------------------------------------------------------------
    # NDVI + TEMPORAL C
    # -------------------------------------------------------------
    def compute_C_factor_precompute(self, ndvi_file, dates):
        NDVI, R = self._read(f"{self.ndvi_dir}/{ndvi_file}")

        # Normalisation comme MATLAB
        NDVI = NDVI / 10000 if NDVI.max() > 1 else NDVI
        NDVI[(NDVI < -1) | (NDVI > 1)] = np.nan

        self.NDVI_baseline = NDVI
        self.ndvi_min = np.nanmin(NDVI)
        self.ndvi_max = np.nanmax(NDVI)

        # Latitude / longitude venant de SpatialData (comme CEQUEAUSpatialData)
        lat = self.spatial.latitude
        lon = self.spatial.longitude

        # >>> ICI : même logique que calculate_regional_bounds.m
        m1, m2, m3, m4 = self._calculate_regional_bounds(lat, lon)

        # Courbe annuelle NDVI (365 jours)
        t = np.arange(1, 366)
        spring = 1 / (1 + np.exp(-m1 * (t - m2)))
        autumn = 1 / (1 + np.exp(m3 * (t - m4)))
        norm_curve = spring + autumn - 1

        self.NDVI_curve_annual = self.ndvi_min + \
            (self.ndvi_max - self.ndvi_min) * norm_curve

        # Cube NDVI temporel
        nDays = len(dates)
        nRows, nCols = NDVI.shape
        NDVI_temporal = np.zeros((nRows, nCols, nDays), dtype=np.float32)

        for i, d in enumerate(dates):
            doy = min(d.timetuple().tm_yday, 365)
            scale = self.NDVI_curve_annual[doy - 1] / self.ndvi_max
            NDVI_t = NDVI * scale
            NDVI_t = np.clip(NDVI_t, self.ndvi_min, self.ndvi_max)
            NDVI_temporal[:, :, i] = NDVI_t

        self.NDVI_temporal = NDVI_temporal

        print("✓ NDVI temporal cube computed")

    # -------------------------------------------------------------
    # REGIONAL PHENOLOGY ENGINE (équivalent calculate_regional_bounds)
    # -------------------------------------------------------------
    def _calculate_regional_bounds(self, lat, lon):
        if (43 <= lat <= 48) and (-68 <= lon <= -59):
            # Maritime Atlantic (PEI/NS/NB)
            m1, m2, m3, m4 = 0.08, 145, 0.05, 285
        elif lat > 58:
            # Subarctic / Territories
            m1, m2, m3, m4 = 0.14, 158, 0.11, 252
        elif (42 <= lat <= 50) and (-90 <= lon <= -74):
            # Great Lakes / Central Canada
            m1, m2, m3, m4 = 0.10, 140, 0.06, 290
        elif (48 <= lat <= 55) and (-115 <= lon <= -94):
            # Prairies
            m1, m2, m3, m4 = 0.12, 135, 0.10, 275
        elif lon < -118:
            # BC / Mountains
            m1, m2, m3, m4 = 0.09, 150, 0.07, 280
        else:
            # Default Central Canada
            lat_factor = max(0, lat - 44)
            m2 = round(125 + lat_factor * 2.5)
            m4 = round(280 - lat_factor * 2.0)
            m1 = 0.08
            m3 = 0.06
        return m1, m2, m3, m4
