import numpy as np
from datetime import datetime, timedelta
from scipy.io import savemat

# Import builders
from spatial.spatial_data import SpatialData
from builders.musle_builder import MUSLEBuilder
from builders.routing_builder import RoutingBuilder
from builders.connectivity_builder import ConnectivityBuilder
from builders.aggregator import Aggregator
from builders.param_dependent_factors import ParamDependentFactors
from builders.ce_pixel_index_builder import CEPixelIndexBuilder


def main():

    print("\n=== CEQUEAU PYTHON PIPELINE START ===\n")

    # --------------------------------------------------------
    # 1. Load spatial data
    # --------------------------------------------------------
    spatial = SpatialData(
        physiography_folder="data/physiography",
        result_folder="data/results"
    )

    # --------------------------------------------------------
    # 2. MUSLE Builder
    # --------------------------------------------------------
    musle = MUSLEBuilder(
        geo_dir="data/physiography",
        soil_dir="data/soil",
        ndvi_dir="data/ndvi",
        lulc_dir="data/lulc",
        output_dir="outputs",
        spatial=spatial
    )

    # Compute MUSLE factors
    #musle.compute_K_factor("sand.tif", "silt.tif", "clay.tif", "orgC.tif")
    #musle.compute_LS_factor("Slope.tif", "FAC.tif", "DEM.tif")
    #musle.compute_P_factor("LULC.tif", "Slope.tif")
    #musle.compute_CFRG_raw("coarse_fragments.tif")  # <-- mets ton vrai fichier ici

    musle.load_K_factor()
    musle.load_LS_factor()
    musle.load_P_factor()
    musle.load_CFRG_factor()

    # NDVI temporal
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(365)]
    musle.compute_C_factor_precompute("NDVI_2024_Landsat.tif", dates)

    # --------------------------------------------------------
    # 3. Routing Builder
    # --------------------------------------------------------
    routing = RoutingBuilder(spatial)

    # --------------------------------------------------------
    # 4. CE pixel index (placeholder)
    # --------------------------------------------------------
    ce_builder = CEPixelIndexBuilder(
        ce_shapefile="data/physiography/CE_fishnet.shp",
        raster_path="data/physiography/DEM.tif"
    )

    CE_pixel_idx = ce_builder.build()

    # --------------------------------------------------------
    # 5. Param-dependent factors
    # --------------------------------------------------------
    params = {
        "MUSLE": {
            "C_alpha": 2.0,
            "C_beta": 0.8,
            "CFRG_exp": 0.05
        },
        "SDR": {
            "alpha": 4.0,
            "beta": -2.0
        },
        "CE_pixel_idx": CE_pixel_idx
    }

    pdf = ParamDependentFactors(musle, params, CE_pixel_idx)
    C_struct = pdf.compute_C_factor()
    musle.C_static = C_struct["C_static"]

    # --------------------------------------------------------
    # 6. Connectivity Builder
    # --------------------------------------------------------
    conn = ConnectivityBuilder(spatial, musle, params)
    conn.compute()

    # --------------------------------------------------------
    # 7. Aggregator
    # --------------------------------------------------------
    agg = Aggregator(spatial, musle, dates, CE_pixel_idx)
    MUSLE_CE = agg.aggregate()


    print("LS_mean stats:", np.nanmin(MUSLE_CE["LS_mean"]), np.nanmax(MUSLE_CE["LS_mean"]))
    print("K_mean stats:", np.nanmin(MUSLE_CE["K_mean"]), np.nanmax(MUSLE_CE["K_mean"]))
    print("P_mode stats:", np.nanmin(MUSLE_CE["P_mode"]), np.nanmax(MUSLE_CE["P_mode"]))
    print("CFRG_CE stats:", np.nanmin(MUSLE_CE["CFRG_CE"]), np.nanmax(MUSLE_CE["CFRG_CE"]))
    print("C_static stats:", np.nanmin(musle.C_static), np.nanmax(musle.C_static))
    print("SDR stats:", np.nanmin(conn.SDR), np.nanmax(conn.SDR))
    print("IC_CE stats:", np.nanmin(conn.IC_CE), np.nanmax(conn.IC_CE))

    # --------------------------------------------------------
    # 8. Save outputs
    # --------------------------------------------------------
    savemat("outputs/MUSLE_inputs.mat", {
        "LS_mean": MUSLE_CE["LS_mean"],
        "K_mean": MUSLE_CE["K_mean"],
        "P_mode": MUSLE_CE["P_mode"],
        "CFRG_CE": MUSLE_CE["CFRG_CE"],
        "C_static": musle.C_static,
        "SDR": conn.SDR,
        "IC_CE": conn.IC_CE,
        "routingOrder": routing.routingOrder
    })

    print("\n=== PIPELINE COMPLETE — FILES SAVED ===\n")


if __name__ == "__main__":
    main()
