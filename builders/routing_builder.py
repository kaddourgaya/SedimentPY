import numpy as np
import pandas as pd


class RoutingBuilder:
    """
    Python equivalent of CEQUEAURoutingBuilder.m
    Computes:
        - CE → CP mapping
        - upstreamCPs
        - downstreamCP
        - routingOrder (topological sort)
        - CP_stream_node
    """

    def __init__(self, spatial):
        self.spatial = spatial

        self.nCE = len(spatial.CEfishnet)
        self.nCP = len(spatial.CPfishnet)
        self.rtable = spatial.rtable

        # Outputs
        self.CE_CP_map = None
        self.upstreamCPs = None
        self.downstreamCP = None
        self.routingOrder = None
        self.CP_stream_node = None

        # Build everything
        self.map_CE_to_CP()
        self.build_routing_order()
        self.map_CP_to_stream()

    # -------------------------------------------------------------
    # CE → CP mapping
    # -------------------------------------------------------------
    def map_CE_to_CP(self):
        CE_CP = []

        CE_ids = self.spatial.CEfishnet["newCEid"].values
        CP_CE_ids = self.spatial.CPfishnet["newCEid"].values
        CP_ids = self.spatial.CPfishnet["newCPid"].values

        for CEid in CE_ids:
            mask = CP_CE_ids == CEid
            CE_CP.append(list(CP_ids[mask]))

        self.CE_CP_map = CE_CP
        print("✓ CE → CP mapping built")

    # -------------------------------------------------------------
    # Build routing order (topological sort)
    # -------------------------------------------------------------
    def build_routing_order(self):
        df = self.rtable

        CP_ids = df["newCPid"].values
        CPid_to_idx = {cpid: i for i, cpid in enumerate(CP_ids)}

        upstream_idx = []
        downstream_idx = []

        for i in range(self.nCP):
            ups_str = df["upstreamCPs"].iloc[i]
            ups = [int(x) for x in ups_str.replace("[", "").replace("]", "").split(",") if x.strip().isdigit()]
            ups = [CPid_to_idx[u] for u in ups if u in CPid_to_idx]
            upstream_idx.append(ups)

            dwn_raw = df["downstreamCPs"].iloc[i]

            # Cas 1 : c’est déjà un entier
            if isinstance(dwn_raw, (int, np.integer)):
                dwn = [int(dwn_raw)]

            # Cas 2 : c’est une chaîne "[12]" ou "[]"
            else:
                s = str(dwn_raw).replace("[", "").replace("]", "")
                dwn = [int(x) for x in s.split(",") if x.strip().isdigit()]

            # Convertir en index
            dwn = [CPid_to_idx[d] for d in dwn if d in CPid_to_idx]
            downstream_idx.append(dwn)


        self.upstreamCPs = upstream_idx
        self.downstreamCP = downstream_idx

        indegree = np.array([len(u) for u in upstream_idx])
        Q = list(np.where(indegree == 0)[0])
        order = []

        while Q:
            cp = Q.pop(0)
            order.append(cp)

            for d in downstream_idx[cp]:
                indegree[d] -= 1
                if indegree[d] == 0:
                    Q.append(d)

        if len(order) != self.nCP:
            raise RuntimeError("Routing graph contains a cycle — routing impossible.")

        self.routingOrder = order
        print("✓ Routing order computed")

    # -------------------------------------------------------------
    # CP → Stream node mapping
    # -------------------------------------------------------------
    def map_CP_to_stream(self):
        CP_ids = self.spatial.CPfishnet["newCPid"].values
        stream = self.spatial.stream_cequeau

        streamCPid = []
        streamMain = []

        for _, row in stream.iterrows():
            name = row.get("names", "")
            cp = None
            if isinstance(name, str) and "CP" in name:
                try:
                    cp = int(name.replace("CP", "").strip())
                except:
                    cp = None
            streamCPid.append(cp)

            main = row.get("main_path", 0)
            streamMain.append(bool(main == 1))

        streamCPid = np.array(streamCPid)
        streamMain = np.array(streamMain)

        CP_stream_node = np.full(self.nCP, np.nan)

        for i in range(self.nCP):
            cpId = CP_ids[i]

            idx_main = np.where((streamCPid == cpId) & (streamMain == True))[0]
            if len(idx_main) > 0:
                CP_stream_node[i] = idx_main[0]
                continue

            ups = self.upstreamCPs[i]
            if len(ups) == 0:
                idx = np.where(streamCPid == cpId)[0]
                if len(idx) > 0:
                    CP_stream_node[i] = idx[0]
                    continue

            dwn = self.downstreamCP[i]
            for d in dwn:
                idx = np.where(streamCPid == CP_ids[d])[0]
                if len(idx) > 0:
                    CP_stream_node[i] = idx[0]
                    break

        self.CP_stream_node = CP_stream_node
        print("✓ CP → Stream node mapping built")
