import numpy as np
import rasterio

class ParamDependentFactors:
    """
    Python equivalent of ParamDependentFactors.m
    Computes:
        - C_temporal (daily C-factor per CE)
        - C_static (from MATLAB C_factor.tif)
        - CFRG_CE (from MATLAB CFRG_factor.tif)
    """

    def __init__(self, musle, params, CE_pixel_idx):
        self.params = params

        # NDVI inputs (déjà calculés dans MUSLEBuilder)
        self.NDVI_temporal = musle.NDVI_temporal
        self.ndvi_min = musle.ndvi_min
        self.ndvi_max = musle.ndvi_max

        # CE pixel index
        self.CE_pixel_idx = CE_pixel_idx

        # ---------------------------------------------------------
        # Charger C_static depuis C_factor.tif (MATLAB)
        # ---------------------------------------------------------
        self.C_static = self._load_raster("data/MUSLE/C_factor.tif")

        # ---------------------------------------------------------
        # Charger CFRG depuis CFRG_factor.tif (MATLAB)
        # ---------------------------------------------------------
        self.CFRG = self._load_raster("data/MUSLE/CFRG_factor.tif")

    # -------------------------------------------------------------
    # Helper pour charger un raster
    # -------------------------------------------------------------
    def _load_raster(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(float)

    # -------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------
    def compute_C_factor(self):
        alpha = self.params["MUSLE"]["C_alpha"]
        beta = self.params["MUSLE"]["C_beta"]

        _, _, nDays = self.NDVI_temporal.shape
        nCE = len(self.CE_pixel_idx)

        C_CE = np.zeros((nDays, nCE), dtype=np.float32)

        # ---------------------------------------------------------
        # DAILY TEMPORAL C (Van der Knijff)
        # ---------------------------------------------------------
        for d in range(nDays):
            NDVI_t = self.NDVI_temporal[:, :, d].copy()

            NDVI_t[NDVI_t >= beta] = beta - 1e-6

            C_t = np.exp(-alpha * (NDVI_t / (beta - NDVI_t)))
            C_t[~np.isfinite(C_t)] = np.nan

            C_CE[d, :] = self._area_weighted_idx(C_t)

        return {
            "C_temporal": C_CE,
            "C_static": self.C_static
        }

    # -------------------------------------------------------------
    # CFRG FACTOR (MATLAB version)
    # -------------------------------------------------------------
    def compute_CFRG_factor(self):
        return self._area_weighted_idx(self.CFRG)

    # -------------------------------------------------------------
    # PRIVATE: AREA-WEIGHTED MEAN PER CE
    # -------------------------------------------------------------
    def _area_weighted_idx(self, raster):
        nCE = len(self.CE_pixel_idx)
        out = np.full(nCE, np.nan)

        for i in range(nCE):
            idx = self.CE_pixel_idx[i]
            rows, cols = np.unravel_index(idx, raster.shape)

            vals = raster[rows, cols]
            vals = vals[~np.isnan(vals)]

            if len(vals) > 0:
                out[i] = np.mean(vals)

        return out
