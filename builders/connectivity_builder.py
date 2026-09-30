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

        # 🔧 nouveaux champs
        self.CE_CP_map = None
        self.CP_stream_node = None

    def compute(self):
        self.load_IC_grid()
        self.aggregate_to_CE()
        self.compute_SDR()

        print("IC_CE stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.nanmin(self.IC_CE), np.nanmax(self.IC_CE), np.nanmean(self.IC_CE)))

        print("SDR stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.nanmin(self.SDR), np.nanmax(self.SDR), np.nanmean(self.SDR)))

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
        self.IC_CE = (ICs - IC_min) / (IC_max - IC_min + 1e-12)

        print("✓ IC_CE aggregated")

    def compute_SDR(self):
        alpha = self.params["SDR"]["alpha"]
        beta = self.params["SDR"]["beta"]
        IC = self.IC_CE
        SDR = 1 / (1 + np.exp(-(alpha * IC + beta)))

        # Étape 1 : éviter les SDR trop proches de 0
        SDR[SDR < 0.05] = 0.05

        # Étape 2 : accentuer les contrastes (gamma > 1)
        gamma = 1.5
        SDR_enh = SDR ** gamma

        # Étape 3 : renormaliser sur [0, 1]
        SDR_min = np.nanmin(SDR_enh)
        SDR_max = np.nanmax(SDR_enh)
        SDR_scaled = (SDR_enh - SDR_min) / (SDR_max - SDR_min + 1e-12)


        self.SDR = np.clip(SDR_scaled, 0, 1)
        print("✓ SDR computed (enhanced contrast)")

    def build_connectivity_from_bassin(self):
        base_results = "data/results"

        # --- carreauxEntiers (CE) ---
        ce_df = pd.read_csv(os.path.join(base_results, "carreauxEntiers.csv"))
        CE_ids = ce_df["CEid"].to_numpy()
        nCE = len(CE_ids)

        # --- carreauxPartiels (CP) ---
        cp_df = pd.read_csv(os.path.join(base_results, "carreauxPartiels.csv"))
        CP_ids = cp_df["CPid"].to_numpy()
        idCE_cp = cp_df["idCE"].to_numpy()
        nCP = len(CP_ids)

        # --- CEtoCP principal ---
        CEtoCP = np.zeros(nCE, dtype=np.int32)
        for i in range(nCE):
            CEid = CE_ids[i]
            candidates = np.where(idCE_cp == CEid)[0]
            if len(candidates) > 0:
                CEtoCP[i] = candidates[0] + 1  # 1‑based
            else:
                CEtoCP[i] = 0
        self.CEtoCP = CEtoCP
        print("✓ CEtoCP built from carreauxPartiels.csv")

        # --- CE_CP_map : tous les CP par CE (1‑based) ---
        CE_CP_map = []
        for i in range(nCE):
            CEid = CE_ids[i]
            candidates = np.where(idCE_cp == CEid)[0]
            # on stocke les indices CP 1‑based
            CE_CP_map.append((candidates + 1).astype(np.int32))
        self.CE_CP_map = CE_CP_map
        print("✓ CE_CP_map built (list of CP per CE)")

        # --- CP_stream_node : pour l’instant tous les CP sur le réseau ---
        # Si tu as une colonne dédiée (ex. "isStream"), tu peux la utiliser ici.
        self.CP_stream_node = np.ones(nCP, dtype=np.int32)
        print("✓ CP_stream_node built (all CP = 1)")

        # --- upstreamCPs reconstruit depuis routing.csv ---
        routing_path = os.path.join(base_results, "routing.csv")
        r_df = pd.read_csv(routing_path)

        if "CPid" not in r_df.columns or "inCPid" not in r_df.columns:
            raise ValueError(
                "routing.csv doit contenir les colonnes 'CPid' et 'inCPid'. "
                f"Colonnes trouvees : {r_df.columns.tolist()}"
            )

        cp_ids = r_df["CPid"].to_numpy(dtype=np.int32)

        # Le C++ attend upstreamCPs[cp - 1] = CPs directement amont de cp.
        upstreamCPs = [np.array([], dtype=np.int32) for _ in range(nCP)]

        for _, row in r_df.iterrows():
            cp_source = int(row["CPid"])
            cp_downstream = int(row["inCPid"])

            # -99999 représente une sortie / valeur sans CP aval valide.
            if cp_downstream < 1 or cp_downstream > nCP:
                continue

            if cp_source < 1 or cp_source > nCP:
                continue

            if cp_source == cp_downstream:
                continue

            # CP IDs sont 1-based ; liste Python est 0-based.
            upstreamCPs[cp_downstream - 1] = np.append(
                upstreamCPs[cp_downstream - 1],
                np.int32(cp_source)
            )

        self.upstreamCPs = upstreamCPs

        n_links = sum(len(x) for x in upstreamCPs)
        n_cp_with_upstream = sum(len(x) > 0 for x in upstreamCPs)

        print(
            f"✓ upstreamCPs reconstruit depuis inCPid : "
            f"{n_links} liens, {n_cp_with_upstream}/{nCP} CP avec amont"
        )

        if 61 <= nCP:
            print(f"DEBUG CP61 upstream = {self.upstreamCPs[60]}")

        # --- routingOrder topologique : têtes de bassin -> exutoire ---
        downstream = {int(cp): [] for cp in cp_ids}
        indegree = {int(cp): 0 for cp in cp_ids}

        for _, row in r_df.iterrows():
            cp_source = int(row["CPid"])
            cp_downstream = int(row["inCPid"])

            if (
                cp_source in downstream
                and cp_downstream in downstream
                and cp_source != cp_downstream
            ):
                downstream[cp_source].append(cp_downstream)
                indegree[cp_downstream] += 1

        queue = sorted([cp for cp in cp_ids if indegree[int(cp)] == 0])
        order = []

        while queue:
            cp = queue.pop(0)
            order.append(cp)

            for cp_downstream in downstream[cp]:
                indegree[cp_downstream] -= 1

                if indegree[cp_downstream] == 0:
                    queue.append(cp_downstream)

        if len(order) != nCP:
            missing = sorted(set(cp_ids.tolist()) - set(order))
            raise RuntimeError(
                "Le graphe routing n'est pas entièrement topologique. "
                f"CP manquants : {missing[:20]}"
            )

        self.routingOrder = np.asarray(order, dtype=np.int32)

        print(
            f"✓ routingOrder topologique construit : "
            f"{len(self.routingOrder)} CP, "
            f"premier={self.routingOrder[0]}, dernier={self.routingOrder[-1]}"
        )

        print("✓ Connectivity (CEtoCP, upstreamCPs, routingOrder, CE_CP_map, CP_stream_node) built — CEQUEAU‑style")
