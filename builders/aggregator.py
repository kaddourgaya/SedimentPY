import numpy as np


class Aggregator:
    """
    Python equivalent of CEQUEAUAggregator.m
    Aggregates MUSLE factors to CE scale:
        - LS_mean
        - K_mean
        - P_mode
        - CFRG_CE
        - C_temporal (via ParamDependentFactors)
    """

    def __init__(self, spatial, musle, dates, CE_pixel_idx):
        self.spatial = spatial
        self.musle = musle
        self.dates = dates
        self.CE_pixel_idx = CE_pixel_idx

        # Outputs
        self.MUSLE_CE = {}

    # -------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------
    def aggregate(self):
        print("\nAggregating MUSLE factors to CE scale...")

        LS = self.musle.LS
        K = self.musle.K
        P = self.musle.P
        CFRG = self.musle.CFRG_raw

        self.MUSLE_CE["LS_mean"] = self._FAC_weighted_mean(LS)
        self.MUSLE_CE["K_mean"] = self._area_weighted_mean(K)

        P_eff, P_mode, mode_frac = self._area_weighted_P_and_mode(P)
        P_final = P_eff.copy()
        mask = (mode_frac >= 0.7) & (~np.isnan(mode_frac))
        P_final[mask] = P_mode[mask]
        self.MUSLE_CE["P_mode"] = P_final

        self.MUSLE_CE["CFRG_CE"] = self._area_weighted_mean(CFRG)

        print("✓ Aggregation complete")
        return self.MUSLE_CE

    # -------------------------------------------------------------
    # FAC-WEIGHTED MEAN (for LS)
    # -------------------------------------------------------------
    def _FAC_weighted_mean(self, raster):
        FAC = self.spatial.FAC
        CE_idx = self.CE_pixel_idx

        nCE = len(CE_idx)
        out = np.full(nCE, np.nan)

        for i in range(nCE):
            idx = CE_idx[i]
            if len(idx) == 0:
                continue

            # Convertir indices aplatis → indices 2D
            rows, cols = np.unravel_index(idx, raster.shape)

            vals = raster[rows, cols]
            w = FAC[rows, cols]

            mask = (~np.isnan(vals)) & (~np.isnan(w)) & (w > 0)
            if np.any(mask):
                out[i] = np.sum(vals[mask] * w[mask]) / np.sum(w[mask])

        return out

    # -------------------------------------------------------------
    # AREA-WEIGHTED MEAN (for K, CFRG)
    # -------------------------------------------------------------
    def _area_weighted_mean(self, raster):
        CE_idx = self.CE_pixel_idx
        nCE = len(CE_idx)
        out = np.full(nCE, np.nan)

        for i in range(nCE):
            idx = CE_idx[i]
            if len(idx) == 0:
                continue

            # Convertir indices aplatis → indices 2D
            rows, cols = np.unravel_index(idx, raster.shape)

            vals = raster[rows, cols]
            vals = vals[~np.isnan(vals)]

            if len(vals) > 0:
                out[i] = np.mean(vals)

        return out

    # -------------------------------------------------------------
    # P FACTOR: MEAN + MODE
    # -------------------------------------------------------------
    def _area_weighted_P_and_mode(self, raster):
        CE_idx = self.CE_pixel_idx
        nCE = len(CE_idx)

        P_eff = np.full(nCE, np.nan)
        P_mode = np.full(nCE, np.nan)
        mode_frac = np.full(nCE, np.nan)

        total_size = raster.size

        for i in range(nCE):
            idx = CE_idx[i]
            if len(idx) == 0:
                continue

            # Ne garder que les indices valides pour ce raster
            idx_valid = idx[idx < total_size]
            if len(idx_valid) == 0:
                continue

            rows, cols = np.unravel_index(idx_valid, raster.shape)
            vals = raster[rows, cols]
            vals = vals[~np.isnan(vals)]

            if len(vals) > 0:
                P_eff[i] = np.mean(vals)

                unique, counts = np.unique(vals, return_counts=True)
                mode_idx = np.argmax(counts)
                P_mode[i] = unique[mode_idx]
                mode_frac[i] = counts[mode_idx] / len(vals)

        return P_eff, P_mode, mode_frac

