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

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    musle.compute_C_factor_precompute("NDVI_2024_Landsat.tif", dates)

    ce_builder = CEPixelIndexBuilder(
        ce_shapefile="data/physiography/CE_fishnet.shp",
        raster_path="data/physiography/FAC.tif"
    )
    CE_pixel_idx = ce_builder.build()

    params = {
        "MUSLE": {
            "C_alpha": 1.0,
            "C_beta": 1.0,
            "CFRG_exp": 5.0
        },
        "SDR": {
            "alpha": 1.0,
            "beta": 1.0
        },
        "CE_pixel_idx": CE_pixel_idx
    }

    pdf = ParamDependentFactors(musle, params, CE_pixel_idx)
    C_struct = pdf.compute_C_factor()

    C_temporal = C_struct["C_temporal"].astype(np.float64)
    C_mean = np.nanmean(C_temporal, axis=0).astype(np.float64)

    conn = ConnectivityBuilder(spatial, musle, params)
    conn.compute()

    agg = Aggregator(spatial, musle, dates, CE_pixel_idx)
    MUSLE_CE = agg.aggregate()

    # Nettoyage CFRG
    CFRG_clean = MUSLE_CE["CFRG_CE"].astype(np.float64)
    nan_mask = np.isnan(CFRG_clean)
    if nan_mask.sum() > 0:
        CFRG_clean[nan_mask] = np.nanmin(CFRG_clean[~nan_mask])

    MUSLE_CE_struct = {
        "LS_mean": MUSLE_CE["LS_mean"].astype(np.float64),
        "K_mean": MUSLE_CE["K_mean"].astype(np.float64),
        "P_mode": MUSLE_CE["P_mode"].astype(np.float64),
        "CFRG_mean": CFRG_clean.astype(np.float64),
        "C_mean": C_mean.astype(np.float64),
        "C_temporal": C_temporal.astype(np.float64),
        "SDR_CE": conn.SDR.astype(np.float64)
    }

    savemat("outputs/MUSLE_inputs.mat", {
        "MUSLE_CE": MUSLE_CE_struct,
        "CEtoCP": conn.CEtoCP.astype(np.int32)
    })

    # upstreamCPs → cell array
    nCP = len(conn.upstreamCPs)
    upstream_cell = np.empty((nCP, 1), dtype=object)
    for i in range(nCP):
        upstream_cell[i, 0] = np.array(conn.upstreamCPs[i], dtype=np.int32)

    # CE_CP_map → cell array
    nCE = len(conn.CE_CP_map)
    CE_CP_cell = np.empty((nCE, 1), dtype=object)
    for i in range(nCE):
        CE_CP_cell[i, 0] = np.array(conn.CE_CP_map[i], dtype=np.int32)

    savemat("outputs/Connectivity_for_CEQUEAU.mat", {
        "routingOrder": conn.routingOrder.astype(np.int32),
        "upstreamCPs": upstream_cell,
        "CEtoCP": conn.CEtoCP.astype(np.int32),
        "CE_CP_map": CE_CP_cell,
        "CP_stream_node": conn.CP_stream_node.astype(np.int32)
    })

    print("\n=== PIPELINE COMPLETE — FILES SAVED ===\n")

if __name__ == "__main__":
    main()
