import numpy as np
from datetime import datetime, timedelta
from scipy.io import savemat

from spatial.spatial_data import SpatialData
from builders.musle_builder import MUSLEBuilder
from builders.connectivity_builder import ConnectivityBuilder
from builders.aggregator import Aggregator
from builders.param_dependent_factors import ParamDependentFactors
from builders.ce_pixel_index_builder import CEPixelIndexBuilder

def main():

    print("\n=== CEQUEAU PYTHON PIPELINE START ===\n")

    spatial = SpatialData(
        physiography_folder="data/physiography",
        result_folder="data/results"
    )

    musle = MUSLEBuilder(
        geo_dir="data/physiography",
        soil_dir="data/soil",
        ndvi_dir="data/ndvi",
        lulc_dir="data/lulc",
        output_dir="outputs",
        spatial=spatial
    )

    musle.load_K_factor()
    musle.load_LS_factor()
    musle.load_P_factor()
    musle.load_CFRG_factor()

    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(365)]
    musle.compute_C_factor_precompute("NDVI_2024_Landsat.tif", dates)

    ce_builder = CEPixelIndexBuilder(
        ce_shapefile="data/physiography/CE_fishnet.shp",
        raster_path="data/physiography/FAC.tif"   # 🔧 grille alignée
    )
    CE_pixel_idx = ce_builder.build()

    params = {
        "MUSLE": {
            "C_alpha": 3.0,
            "C_beta": 0.8,
            "CFRG_exp": 0.1
        },
        "SDR": {
            "alpha": 5.0,
            "beta": -2.0
        },
        "CE_pixel_idx": CE_pixel_idx
    }

    pdf = ParamDependentFactors(musle, params, CE_pixel_idx)
    C_struct = pdf.compute_C_factor()

    C_temporal = C_struct["C_temporal"].astype(np.float64)   # [T, CE]
    C_mean = np.nanmean(C_temporal, axis=0).astype(np.float64)

    conn = ConnectivityBuilder(spatial, musle, params)
    conn.compute()

    agg = Aggregator(spatial, musle, dates, CE_pixel_idx)
    MUSLE_CE = agg.aggregate()

    routingOrder = conn.CEtoCP.astype(np.int32)

    # Nettoyage CFRG
    CFRG_clean = MUSLE_CE["CFRG_CE"].astype(np.float64)
    nan_mask = np.isnan(CFRG_clean)

    print(f"\n=== CFRG_CE CLEANUP ===")
    print(f"NaN détectés : {nan_mask.sum()}")

    if nan_mask.sum() > 0:
        valid = CFRG_clean[~nan_mask]
        min_valid = np.nanmin(valid)
        CFRG_clean[nan_mask] = min_valid

    print("Nettoyage CFRG terminé.")
    print(f"min={CFRG_clean.min():.6f}  max={CFRG_clean.max():.6f}\n")

    savemat("outputs/MUSLE_inputs.mat", {
        "LS_mean": MUSLE_CE["LS_mean"].astype(np.float64),
        "K_mean": MUSLE_CE["K_mean"].astype(np.float64),
        "P_mode": MUSLE_CE["P_mode"].astype(np.float64),
        "CFRG_CE": CFRG_clean,
        "C_static": C_mean.astype(np.float64),
        "C_temporal": C_temporal.reshape(-1, order="C"),
        "SDR": conn.SDR.astype(np.float64).reshape(-1, order="C"),
        "IC_CE": conn.IC_CE.astype(np.float64),
        "routingOrder": routingOrder
    })

    savemat("outputs/Connectivity_for_CEQUEAU.mat", {
        "routingOrder": routingOrder,
        "upstreamCPs": np.array(conn.upstreamCPs, dtype=object),
        "CEtoCP": conn.CEtoCP.astype(np.int32)
    })

    print("\n=== PIPELINE COMPLETE — FILES SAVED ===\n")

if __name__ == "__main__":
    main()
