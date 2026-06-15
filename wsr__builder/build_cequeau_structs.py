import numpy as np
import json
import os
from scipy.io import savemat, loadmat
from netCDF4 import Dataset
import datetime as dt

def to_struct_array(list_of_dicts):
    fields = list(list_of_dicts[0].keys())
    dtype = [(f, object) for f in fields]
    arr = np.zeros(len(list_of_dicts), dtype=dtype)
    for i, d in enumerate(list_of_dicts):
        for f in fields:
            arr[f][i] = d[f]
    return arr

# ------------------------------------------------------------
# 2) BASSIN VERSANT
# ------------------------------------------------------------
def load_bassin_versant(json_path):
    with open(json_path, "r") as f:
        bv_json = json.load(f)

    ce = bv_json["carreauxEntiers"]
    cp = bv_json["carreauxPartiels"]

    # --- CE ---
    CEarray = []
    for i in range(len(ce["CEid"])):
        entry = {}
        for k, v in ce.items():
            val = v[i] if isinstance(v, list) else v
            entry[k] = float(val) if isinstance(val, (int, float)) else val
        CEarray.append(entry)

    # --- CP ---
    CParray = []
    for i in range(len(cp["CPid"])):
        entry = {}
        for k, v in cp.items():
            val = v[i] if isinstance(v, list) else v

            if isinstance(val, (int, float)):
                entry[k] = float(val)
            elif isinstance(val, list):
                flat = []
                for x in val:
                    if isinstance(x, list):
                        flat.extend(x)
                    else:
                        flat.append(x)
                entry[k] = np.array(flat, dtype=float)
            else:
                entry[k] = val

        # CEQUEAU: garder la même logique que la collègue
        entry["CPid"] = float(cp["CPid"][i])
        entry["idCP"] = entry["CPid"]

        # NE PLUS MODIFIER idCPAval, idCPsAmont, penteRiviere ici
        # -> on laisse les valeurs du JSON telles quelles

        CParray.append(entry)

    return CEarray, CParray

# ------------------------------------------------------------
# 3) PARAMETRES
# ------------------------------------------------------------
def load_parametres(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------
# 4) METEO
# ------------------------------------------------------------
def load_meteo(nc_path):
    nc = Dataset(nc_path)

    tMin = nc["tMin"][:]
    tMax = nc["tMax"][:]
    t    = (tMin + tMax) / 2.0

    meteo = {
        "pTot":        nc["pTot"][:].astype(np.single),
        "tMin":        tMin.astype(np.single),
        "tMax":        tMax.astype(np.single),
        "rayonnement": nc["rayonnement"][:].astype(np.single),
        "nebulosite":  nc["nebulosite"][:].astype(np.single),
        "pression":    nc["pression"][:].astype(np.single),
        "vitesseVent": nc["vitesseVent"][:].astype(np.single),
        "t":           t.astype(np.single),
    }

    def datenum(y, m, d):
        return (dt.datetime(y, m, d) - dt.datetime(1, 1, 1)).days + 366

    execution = {
        "dateDebut": float(datenum(1978, 12, 31)),
        "dateFin":   float(datenum(2025, 12, 28)),
    }

    return meteo, execution

# ------------------------------------------------------------
# 5) CONNECTIVITÉ
# ------------------------------------------------------------
def build_connectivity(json_path, musle_path, out_path):
    with open(json_path, "r") as f:
        bv = json.load(f)

    cp = bv["carreauxPartiels"]
    ce = bv["carreauxEntiers"]

    nCP = len(cp["CPid"])
    nCE = len(ce["CEid"])

    cp_ids = [float(x) for x in cp["CPid"]]
    id_to_idx = {cp_ids[i]: i for i in range(nCP)}

    upstream = []
    for i in range(nCP):
        raw = cp["idCPsAmont"][i]
        flat = []
        for x in raw:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        idxs = []
        for cid in flat:
            cid = float(cid)
            if cid != 0:
                idxs.append(id_to_idx[cid] + 1)
        upstream.append(np.array(idxs, dtype=np.int32))

    CEtoCP = np.zeros(nCE, dtype=np.int32)
    for i in range(nCE):
        CEid = float(ce["CEid"][i])
        candidates = [j for j in range(nCP) if float(cp["idCE"][j]) == CEid]
        CEtoCP[i] = candidates[0] + 1 if candidates else 0

    musle = loadmat(musle_path)
    routingOrder = musle["routingOrder"].astype(np.int32).ravel()

    savemat(out_path, {
        "routingOrder": routingOrder,
        "upstreamCPs": np.array(upstream, dtype=object),
        "CEtoCP": CEtoCP
    })

# ------------------------------------------------------------
# 6) MAIN
# ------------------------------------------------------------
def build_all():
    base_json = r"D:\CEQUEAU-Sed\data\results"
    out       = r"C:\Stage\CEQUEAU\data\rebuilt_mat"
    os.makedirs(out, exist_ok=True)

    # Charger CE/CP depuis JSON
    CEarray, CParray = load_bassin_versant(os.path.join(base_json, "bassinVersant.json"))
    nCE = len(CEarray)

    # Charger les surfaces CEQUEAU originales
    surf_ref = loadmat(r"C:\Stage\CEQUEAU\surfaces_reference.mat")
    superficieCE = surf_ref["surfCE_ref"].ravel()

    # Recalcul superficieCP cohérente
    superficieCP = np.zeros(len(CParray))
    for i, ce in enumerate(CEarray):
        CEid = ce["CEid"]
        idx = [j for j, cpj in enumerate(CParray) if cpj["idCE"] == CEid]
        if len(idx) == 0:
            continue
        pct = np.array([CParray[j]["pctSurface"] for j in idx], dtype=float)
        pct_norm = pct / pct.sum()
        superficieCP[idx] = pct_norm * superficieCE[i]

    # >>> IMPORTANT : récupérer nbCpCheminLong depuis le WSRStruct de référence <<<
    ref = loadmat(r"D:\CEQUEAU-Sed\data\Surface_Runoff\WSRStruct2025.mat")
    nbCpCheminLong_ref = float(ref["WSRStruct"]["bassinVersant"][0,0]["nbCpCheminLong"])

    # Construire bassinVersant
    bassinVersant = {
        "nomBassinVersant": "Rebuilt_BV",
        "barrage": np.zeros((0,)),
        "nbCpCheminLong": nbCpCheminLong_ref,   # <<< au lieu de 0.0
        "carreauxEntiers": to_struct_array(CEarray),
        "carreauxPartiels": to_struct_array(CParray),
        "superficieCE": superficieCE,
        "superficieCP": superficieCP,
        "superficieTotale": float(superficieCP.sum()),
        "superficieBassin": float(superficieCE.sum())
    }

    savemat(os.path.join(out, "bassinVersant_CEQUEAU.mat"), {"bassinVersant": bassinVersant})

    # Paramètres
    param = load_parametres(os.path.join(base_json, "parameters.json"))

    def expand(v):
        if isinstance(v, list) or isinstance(v, np.ndarray):
            return np.array(v, dtype=float)
        else:
            return np.full(nCE, float(v), dtype=float)

    sol = param["sol"]
    for key in ["cin_s","cvnb_s","cvnh_s","cvsi_s",
                "hinf_s","hint_s","hnap_s","hpot_s",
                "hsol_s","hrimp_s","tri_s"]:
        sol[key] = expand(sol[key])

    param["surface"] = {
        "superficieCE": superficieCE,
        "superficieCP": superficieCP,
        "superficieTotale": float(superficieCP.sum()),
        "superficieBassin": float(superficieCE.sum())
    }

    param["tc"] = {
        "tau_erosion": 0.009,
        "tau_deposition": 1.50,
        "shearscale": 0.04,
        "shearexpo": 0.75,
        "capacity_coeff": 0.08,
        "Background_Supply": 0.05,
        "Background_Wash": 15.00 # 1.5, 2.5
    }


    savemat(os.path.join(out, "parameters.mat"), {"parameters": param})

    # Connectivité
    build_connectivity(
        os.path.join(base_json, "bassinVersant.json"),
        r"D:\SedimentPY\outputs\MUSLE_inputs.mat",
        os.path.join(out, "Connectivity_for_CEQUEAU.mat")
    )

    # Météo
    meteo, execution = load_meteo(
        r"C:\Users\jugur\Downloads\pycequeau\pycequeau\meteo\meteo_cequeau.nc"
    )

    param["sol"] = sol

    WSR = {
        "bassinVersant": bassinVersant,
        "parametres": param,
        "meteoInterpolee": meteo,
        "execution": execution
    }

    savemat(os.path.join(out, "WSRStruct_light.mat"), {"WSRStruct": WSR})

    print("\n=== STRUCTURES CEQUEAU RECONSTRUITES (alignées sur WSRStruct2025) ===\n")

if __name__ == "__main__":
    build_all()
