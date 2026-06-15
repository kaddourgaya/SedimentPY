import numpy as np


class SedimentRoutingWashload:
    """
    Python equivalent of CEQUEAUSedimentRouting_Washload
    - CE-scale SY_CE → CP via SDR and CE_CP_map
    - In-channel routing with separate erosion/deposition shear thresholds
      and background washload / background supply
    - Produces SY_routed (t/day), SSC_routed (mg/L) for all CPs,
      and SSC_outlet (mg/L) at outlet CP.
    """

    def __init__(self, project, params):
        """
        Parameters
        ----------
        project : dict-like
            Must contain:
              - 'SY_CE'          : (nDays, nCE) daily sediment yield at CE scale (t/day)
              - 'CE_CP_map'      : list of lists, CE → CP indices (0-based)
              - 'CP_area'        : (nCP,) CP contributing area [m²] or [km²] (see note)
              - 'SDR'            : (nCE,) or (nDays, nCE) SDR values
              - 'Flow'           : (nDays, nCP) discharge (m³/s)
              - 'upstreamCPs'    : list of lists of upstream CP indices (0-based)
              - 'downstreamCP'   : list of lists of downstream CP indices (0-based)
              - 'order'          : (nCP,) CP routing order (headwater → outlet)
              - 'CP_stream_node' : (nCP,) stream node index or NaN for hillslope-only
              - 'Outlet' (optional): outlet CP index (0-based); if absent, we infer.
        params : dict-like
            Must contain:
              - 'TC' with keys:
                  * 'tau_erosion'
                  * 'tau_deposition'
                  * 'shearscale'
                  * 'shearexpo'
                  * 'capacity_coeff'
                  * 'Background_Supply'
                  * 'Background_Wash'
        """
        self.SY_CE = np.asarray(project["SY_CE"], dtype=float)  # (nDays, nCE)
        self.CE_CP_map = project["CE_CP_map"]
        self.CP_area = np.asarray(project["CP_area"], dtype=float)  # same unit for all CPs
        self.SDR = np.asarray(project["SDR"], dtype=float)
        self.Flow = np.asarray(project["Flow"], dtype=float)  # (nDays, nCP)

        self.upstreamCPs = project["upstreamCPs"]
        self.downstreamCP = project["downstreamCP"]
        self.order = np.asarray(project["order"], dtype=int)
        self.CP_stream_node = np.asarray(project["CP_stream_node"], dtype=float)

        self.nDays, self.nCE = self.SY_CE.shape
        self.nCP = self.Flow.shape[1]

        # Stream vs hillslope CPs
        self.isStreamCP = ~np.isnan(self.CP_stream_node) & (self.CP_stream_node > 0)
        self.isHillslopeCP = ~self.isStreamCP

        # Outlet CP index (0-based)
        if "Outlet" in project:
            self.Outlet = int(project["Outlet"])
        else:
            # Fallback: last CP in routing order that is stream-connected and has no downstream
            self.Outlet = self._infer_outlet()

        # TC parameters
        tc = params["TC"]
        self.tau_erosion = float(tc["tau_erosion"])
        self.tau_deposition = float(tc["tau_deposition"])
        self.shearscale = float(tc["shearscale"])
        self.shearexpo = float(tc["shearexpo"])
        self.capacity_coeff = float(tc["capacity_coeff"])
        self.bg_supply = float(tc["Background_Supply"])
        self.bg_wash = float(tc["Background_Wash"])

        # Outputs
        self.SY_CP = np.zeros((self.nDays, self.nCP), dtype=float)       # hillslope → CP
        self.SY_routed = np.zeros((self.nDays, self.nCP), dtype=float)   # routed in-channel
        self.local_bed_storage = np.zeros((self.nDays, self.nCP), dtype=float)
        self.SSC_routed = np.full((self.nDays, self.nCP), np.nan, dtype=float)
        self.SSC_outlet = np.full(self.nDays, np.nan, dtype=float)

        # Run model
        self.transfer_hillslope()
        self.route_stream_sed()

    # ------------------------------------------------------------------
    # Helper: infer outlet CP if not provided
    # ------------------------------------------------------------------
    def _infer_outlet(self):
        """
        Choose a stream CP with no downstream CPs as outlet.
        If multiple, pick the one with largest CP index.
        """
        candidates = []
        for cp in range(self.nCP):
            if not self.isStreamCP[cp]:
                continue
            ds = [d for d in self.downstreamCP[cp] if 0 <= d < self.nCP]
            if len(ds) == 0:
                candidates.append(cp)
        if not candidates:
            # fallback: last stream CP in order
            stream_in_order = [cp for cp in self.order if self.isStreamCP[cp]]
            if not stream_in_order:
                return 0
            return int(stream_in_order[-1])
        return int(max(candidates))

    # ------------------------------------------------------------------
    # Step 1: CE → CP transfer via SDR
    # ------------------------------------------------------------------
    def transfer_hillslope(self):
        """
        Compute SY_CP (t/day) at each CP from CE-scale SY_CE and SDR.
        Only stream CPs receive sediment (like MATLAB version).
        """
        # SDR can be (nCE,) or (nDays, nCE)
        if self.SDR.ndim == 1:
            SDR_daily = np.tile(self.SDR[None, :], (self.nDays, 1))
        else:
            SDR_daily = self.SDR

        for iCE in range(self.nCE):
            cpids = self.CE_CP_map[iCE]
            if not cpids:
                continue

            cpids = np.asarray(cpids, dtype=int)
            # Keep only stream CPs
            mask_stream = self.isStreamCP[cpids]
            stream_cpids = cpids[mask_stream]
            if stream_cpids.size == 0:
                continue

            # Area fraction among stream CPs
            A_stream = np.sum(self.CP_area[stream_cpids])
            if A_stream <= 0:
                continue

            frac = self.CP_area[stream_cpids] / A_stream  # (n_stream_cp,)

            # Daily CE yield * SDR
            Yield = self.SY_CE[:, iCE] * SDR_daily[:, iCE]  # (nDays,)

            # Broadcast to CPs
            self.SY_CP[:, stream_cpids] += Yield[:, None] * frac[None, :]

    # ------------------------------------------------------------------
    # Step 2: In-channel routing with erosion/deposition + washload
    # ------------------------------------------------------------------
    def route_stream_sed(self):
        """
        Two-threshold energy state routing:
        - tau_erosion: above → erosion + possible resuspension
        - tau_deposition: below → deposition
        - between → equilibrium transport with background washload
        """
        SY_route = np.zeros((self.nDays, self.nCP), dtype=float)
        mean_area = np.mean(self.CP_area[self.isStreamCP]) if np.any(self.isStreamCP) else 1.0

        for t in range(self.nDays):
            # propagate bed storage from previous day
            if t > 0:
                self.local_bed_storage[t, :] = self.local_bed_storage[t - 1, :]
            else:
                self.local_bed_storage[t, :] = 0.0

            for cp in self.order:
                if not self.isStreamCP[cp]:
                    continue

                # --- A. Upstream sediment ---
                up = [u for u in self.upstreamCPs[cp] if 0 <= u < self.nCP]
                up = [u for u in up if self.isStreamCP[u]]
                if len(up) == 0:
                    Sed_up = 0.0
                else:
                    Sed_up = np.sum(SY_route[t, up])

                Sed_local = self.SY_CP[t, cp]
                Sed_in = Sed_up + Sed_local

                # --- B. Hydraulic stress ---
                Q = self.Flow[t, cp]
                if not np.isfinite(Q) or Q < 0:
                    Q = 0.0

                tau_0 = self.shearscale * (Q ** self.shearexpo)
                Current_Bed = self.local_bed_storage[t, cp]

                # Background washload (mg/L baseline → t/day)
                if Q > 0:
                    # same structure as MATLAB: factor * Q * 0.0864
                    Wash_Factor = self.bg_wash * Q
                    Background_Wash = Wash_Factor * Q * 0.0864
                else:
                    Background_Wash = 0.0

                # --- C. Energy zones ---
                if tau_0 > self.tau_erosion:
                    # Zone 1: High energy (erosion / hungry flow)
                    Area_Scale_Factor = self.CP_area[cp] / mean_area
                    TC = self.capacity_coeff * Area_Scale_Factor * (tau_0 - self.tau_erosion) ** 1.5

                    if Sed_in > TC:
                        # inflow exceeds capacity → forced transport
                        Sed_move = Sed_in
                        self.local_bed_storage[t, cp] = Current_Bed
                    else:
                        # flow can erode extra material
                        Erosion_Demand = TC - Sed_in
                        Background_Supply = self.bg_supply * (tau_0 - self.tau_erosion)

                        Available_Sediment = Current_Bed + Background_Supply
                        Actual_Resuspension = min(Erosion_Demand, Available_Sediment)

                        Sed_move = Sed_in + Actual_Resuspension

                        Loose_Bed_Removed = max(0.0, Actual_Resuspension - Background_Supply)
                        self.local_bed_storage[t, cp] = max(0.0, Current_Bed - Loose_Bed_Removed)

                elif tau_0 < self.tau_deposition:
                    # Zone 2: Low energy (deposition)
                    if self.tau_deposition > 0:
                        settling_eff = 1.0 - (tau_0 / self.tau_deposition)
                        settling_eff = max(0.0, min(1.0, settling_eff))
                    else:
                        settling_eff = 1.0

                    Actual_Deposition = Sed_in * settling_eff
                    Sed_after_settling = max(0.0, Sed_in - Actual_Deposition)

                    Sed_move = max(Background_Wash, Sed_after_settling)

                    # bed storage gains what is left behind
                    self.local_bed_storage[t, cp] = Current_Bed + (Sed_in - Sed_after_settling)

                else:
                    # Zone 3: Equilibrium transport
                    Sed_move = max(Background_Wash, Sed_in)
                    self.local_bed_storage[t, cp] = Current_Bed

                # --- D. Store routed mass and SSC at CP ---
                SY_route[t, cp] = Sed_move

                if Q > 0:
                    # t/day → mg/L: Sed_move / (Q * 0.0864)
                    self.SSC_routed[t, cp] = Sed_move / (Q * 0.0864)
                else:
                    self.SSC_routed[t, cp] = np.nan

            # Outlet SSC
            Q_out = self.Flow[t, self.Outlet]
            if Q_out > 0:
                self.SSC_outlet[t] = SY_route[t, self.Outlet] / (Q_out * 0.0864)
            else:
                self.SSC_outlet[t] = np.nan

        self.SY_routed = SY_route
