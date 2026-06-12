import numpy as np
import rasterio

class ConnectivityBuilder:
    """
    Python equivalent of CEQUEAUConnectivityBuilder.m
    Computes:
        - IC_pixel (from IC_grid.tif)
        - IC_CE (CE-level IC)
        - SDR (Sediment Delivery Ratio)
    """

    def __init__(self, spatial, musle, params):
        self.spatial = spatial
        self.musle = musle
        self.params = params

        # Outputs
        self.IC_pixel = None
        self.IC_CE = None
        self.SDR = None

        # CE pixel indices
        self.CE_pixel_idx = params.get("CE_pixel_idx", None)
        if self.CE_pixel_idx is None:
            raise ValueError("CE_pixel_idx must be provided in params.")

    # -------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------
    def compute(self):
        self.load_IC_grid()
        self.aggregate_to_CE()
        self.compute_SDR()

    # -------------------------------------------------------------
    # LOAD IC_grid.tif (MATLAB version)
    # -------------------------------------------------------------
    def load_IC_grid(self):
        with rasterio.open("data/MUSLE/IC_grid.tif") as src:
            self.IC_pixel = src.read(1).astype(float)

        # Replace invalid values
        self.IC_pixel[self.IC_pixel < -999] = np.nan
        print("✓ IC_pixel loaded from IC_grid.tif")

    # -------------------------------------------------------------
    # AGGREGATE IC TO CE SCALE (FA-weighted mean)
    # -------------------------------------------------------------
    def aggregate_to_CE(self):
        IC = self.IC_pixel
        FAC = self.spatial.FAC
        CE_idx = self.CE_pixel_idx

        nCE = len(CE_idx)
        ICs = np.full(nCE, np.nan)

        for i in range(nCE):
            idx = CE_idx[i]
            if len(idx) == 0:
                continue

            rows, cols = np.unravel_index(idx, IC.shape)

            vals = IC[rows, cols]
            w = FAC[rows, cols]

            mask = (~np.isnan(vals)) & (w > 0)
            if np.any(mask):
                ICs[i] = np.sum(vals[mask] * w[mask]) / np.sum(w[mask])

        # Normalize 0–1
        IC_min = np.nanmin(ICs)
        IC_max = np.nanmax(ICs)
        self.IC_CE = (ICs - IC_min) / (IC_max - IC_min)

        print("✓ IC_CE aggregated")

    # -------------------------------------------------------------
    # STRUCTURAL SDR
    # -------------------------------------------------------------
    def compute_SDR(self):
        alpha = self.params["SDR"]["alpha"]
        beta = self.params["SDR"]["beta"]

        IC = self.IC_CE
        SDR = 1 / (1 + np.exp(-(alpha * IC + beta)))
        SDR = np.clip(SDR, 0, 1)

        self.SDR = SDR
        print("✓ SDR computed")
