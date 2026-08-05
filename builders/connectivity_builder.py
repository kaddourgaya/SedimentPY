import numpy as np
import rasterio
import pandas as pd
import os

class ConnectivityBuilder:
    def __init__(self, spatial, musle, params):
        self.spatial = spatial
        self.musle = musle
        self.params = params

        self.IC_pixel = None
        self.IC_CE = None
        self.SDR = None

        self.CE_pixel_idx = params["CE_pixel_idx"]

        self.CEtoCP = None
        self.upstreamCPs = None
        self.routingOrder = None

    def compute(self):
        self.load_IC_grid()
        self.aggregate_to_CE()
        self.compute_SDR()
        self.build_connectivity_from_bassin()

    def load_IC_grid(self):
        with rasterio.open("data/MUSLE/IC_grid.tif") as src:
            self.IC_pixel = src.read(1).astype(float)
        self.IC_pixel[self.IC_pixel < -999] = np.nan
        print("✓ IC_pixel loaded")

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

        IC_min = np.nanmin(ICs)
        IC_max = np.nanmax(ICs)
        self.IC_CE = (ICs - IC_min) / (IC_max - IC_min + 1e-12)  # évite division par 0

        print("✓ IC_CE aggregated")

    def compute_SDR(self):
        alpha = self.params["SDR"]["alpha"]
        beta = self.params["SDR"]["beta"]
        IC = self.IC_CE
        SDR = 1 / (1 + np.exp(-(alpha * IC + beta)))
        self.SDR = np.clip(SDR, 0, 1)
        print("✓ SDR computed")

    # ---------------------------------------------------------
    # CEQUEAU connectivity from data/results/*.csv
    # ---------------------------------------------------------
    def build_connectivity_from_bassin(self):
        base_results = "data/results"

        # --- carreauxEntiers (CE) ---
        ce_df = pd.read_csv(os.path.join(base_results, "carreauxEntiers.csv"))
        # colonnes typiques : CEid, x, y, ...
        CE_ids = ce_df["CEid"].to_numpy()
        nCE = len(CE_ids)

        # --- carreauxPartiels (CP) ---
        cp_df = pd.read_csv(os.path.join(base_results, "carreauxPartiels.csv"))
        # colonnes typiques : CPid, idCE, pctSurface, ...
        CP_ids = cp_df["CPid"].to_numpy()
        idCE_cp = cp_df["idCE"].to_numpy()
        nCP = len(CP_ids)

        # --- CEtoCP : même logique que CEQUEAU ---
        CEtoCP = np.zeros(nCE, dtype=np.int32)
        for i in range(nCE):
            CEid = CE_ids[i]
            # tous les CP qui ont ce CEid
            candidates = np.where(idCE_cp == CEid)[0]
            if len(candidates) > 0:
                # CEQUEAU utilise un CP principal (premier)
                CEtoCP[i] = candidates[0] + 1  # CP index 1‑based
            else:
                CEtoCP[i] = 0

        self.CEtoCP = CEtoCP
        print("✓ CEtoCP built from carreauxPartiels.csv")

        # --- upstreamCPs reconstruits depuis routing.csv ---
        routing_path = os.path.join(base_results, "routing.csv")
        r_df = pd.read_csv(routing_path)

        # colonne CPid
        cp_ids = r_df["CPid"].to_numpy()

        # colonne CP_amont (séparée par virgules)
        upstreamCPs = []
        for i in range(len(cp_ids)):
            if "CP_amont" in r_df.columns:
                raw = str(r_df.iloc[i]["CP_amont"])
                if raw.strip() == "" or raw.lower() == "nan":
                    upstreamCPs.append(np.array([], dtype=np.int32))
                else:
                    vals = [int(x) for x in raw.split(",") if x.strip().isdigit()]
                    upstreamCPs.append(np.array(vals, dtype=np.int32))
            else:
                upstreamCPs.append(np.array([], dtype=np.int32))

        self.upstreamCPs = upstreamCPs
        print("✓ upstreamCPs rebuilt from routing.csv")

        # --- routingOrder : soit routing.csv, soit CEtoCP ---
        routing_path = os.path.join(base_results, "routing.csv")
        if os.path.exists(routing_path):
            r_df = pd.read_csv(routing_path)
            # colonne typique : CPid ou ordre
            if "CPid" in r_df.columns:
                self.routingOrder = r_df["CPid"].to_numpy().astype(np.int32)
            else:
                self.routingOrder = r_df.iloc[:, 0].to_numpy().astype(np.int32)
            print("✓ routingOrder loaded from routing.csv")
        else:
            # fallback : CEtoCP (comme tu faisais)
            self.routingOrder = self.CEtoCP.copy()
            print("✓ routingOrder = CEtoCP (fallback)")

        print("✓ Connectivity (CEtoCP, upstreamCPs, routingOrder) built — CEQUEAU‑style")
