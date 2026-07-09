import numpy as np
import rasterio

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
        self.IC_CE = (ICs - IC_min) / (IC_max - IC_min)

        print("✓ IC_CE aggregated")

    def compute_SDR(self):
        alpha = self.params["SDR"]["alpha"]
        beta = self.params["SDR"]["beta"]
        IC = self.IC_CE
        SDR = 1 / (1 + np.exp(-(alpha * IC + beta)))
        self.SDR = np.clip(SDR, 0, 1)
        print("✓ SDR computed")

    # ---------------------------------------------------------
    # CEQUEAU connectivity from bassinVersant.json
    # ---------------------------------------------------------
    def build_connectivity_from_bassin(self):
        CE = self.spatial.CE
        CP = self.spatial.CP

        nCE = len(CE["CEid"])
        nCP = len(CP["CPid"])

        # CEtoCP
        CEtoCP = np.zeros(nCE, dtype=np.int32)
        idCE_cp = [float(x) for x in CP["idCE"]]
        cp_ids  = [float(x) for x in CP["CPid"]]

        for i in range(nCE):
            CEid = float(CE["CEid"][i])
            candidates = [j for j in range(nCP) if idCE_cp[j] == CEid]
            CEtoCP[i] = candidates[0] + 1 if candidates else 0

        self.CEtoCP = CEtoCP
        print("✓ CEtoCP built")

        # upstreamCPs
        upstream = []
        for i in range(nCP):
            raw = CP["idCPsAmont"][i]
            flat = []
            for x in raw:
                flat.append(int(x))
            cleaned = [x for x in flat if x != 0]
            upstream.append(np.array(cleaned, dtype=np.int32))

        self.upstreamCPs = upstream
        print("✓ upstreamCPs built")

        # routingOrder = CEtoCP (CEQUEAU expects CE→CP mapping)
        self.routingOrder = self.CEtoCP.copy()
        print("✓ routingOrder = CEtoCP (CEQUEAU-compatible)")

