import numpy as np
import datetime as dt
import json
import os
from scipy.io import savemat
from scipy.io import loadmat
from netCDF4 import Dataset
import pandas as pd

def to_struct_array(list_of_dicts):
    fields = list(list_of_dicts[0].keys())
    dtype = [(f, object) for f in fields]
    arr = np.zeros(len(list_of_dicts), dtype=dtype)
    for i, d in enumerate(list_of_dicts):
        for f in fields:
            arr[f][i] = d[f]
    return arr

def load_bassin_versant(base_path):
    with open(os.path.join(base_path, "bassinVersant.json"), "r") as f:
        bv_json = json.load(f)

    ce = bv_json["carreauxEntiers"]
    cp = bv_json["carreauxPartiels"]

    # --- CE ---
    nCE = len(ce["CEid"])
    CEarray = []
    for i in range(nCE):
        entry = {}
        for k, v in ce.items():
            val = v[i] if isinstance(v, list) else v
            if isinstance(val, (int, np.integer)):
                val = float(val)
            entry[k] = val
        CEarray.append(entry)

    # --- CP ---
    nCP = len(cp["CPid"])
    CParray = []
    for i in range(nCP):
        entry = {}
        for k, v in cp.items():
            val = v[i] if isinstance(v, list) else v
            if isinstance(val, (int, np.integer)):
                val = float(val)
            entry[k] = val

        entry["CPid"] = float(cp["CPid"][i])
        entry["idCP"] = entry["CPid"]

        if entry["idCPAval"] == 0:
            entry["idCPAval"] = entry["idCP"]

        raw = cp["idCPsAmont"][i]
        flat = []
        for x in raw:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        flat = [float(x if x != 0 else entry["idCP"]) for x in flat]
        entry["idCPsAmont"] = np.array(flat, dtype=float)

        if entry["penteRiviere"] > 100:
            entry["penteRiviere"] = 100.0
        entry["penteRiviere"] = float(entry["penteRiviere"])

        CParray.append(entry)

    bassinVersant = {
        "nomBassinVersant": "Rebuilt_BV",
        "barrage": np.zeros((0,)),
        "nbCpCheminLong": 0.0,
        "carreauxEntiers": to_struct_array(CEarray),
        "carreauxPartiels": to_struct_array(CParray)
    }

    return bassinVersant, CEarray, CParray

def load_parametres(base_path):
    with open(os.path.join(base_path, "parameters.json"), "r") as f:
        return json.load(f)

def apply_ruiss_parameters_from_excel(parametres):
    excel_path = r"D:\CEQUEAU-Sed\data\Surface_Runoff\Parameters_Ruiss.xlsx"
    df = pd.read_excel(excel_path, header=None)
    vals = df.iloc[:, 0].to_numpy().astype(float)

    if len(vals) < 25:
        raise RuntimeError("Parameters_Ruiss.xlsx doit contenir au moins 25 valeurs (sol + transfert + fonte + evapo).")

    parametres.setdefault("sol", {})
    sol = parametres["sol"]

    # 1–14 : sol
    sol["cin_s"]   = float(vals[0])
    sol["cvmar"]   = float(vals[1])
    sol["cvnb_s"]  = float(vals[2])
    sol["cvnh_s"]  = float(vals[3])
    sol["cvsb"]    = float(vals[4])
    sol["cvsi_s"]  = float(vals[5])
    sol["xinfma"]  = float(vals[6])
    sol["hinf_s"]  = float(vals[7])
    sol["hint_s"]  = float(vals[8])
    sol["hmar"]    = float(vals[9])
    sol["hnap_s"]  = float(vals[10])
    sol["hpot_s"]  = float(vals[11])
    sol["hsol_s"]  = float(vals[12])
    sol["hrimp_s"] = float(vals[13])
    # 15 : tri_s dans ton Excel, mais en MATLAB elle l’utilise comme exxkt
    # donc on suit EXACTEMENT son mapping :
    sol["tri_s"]   = float(vals[14])  # si tu veux garder l’info dans sol aussi

    # 15 : transfert.exxkt
    parametres.setdefault("transfert", {})
    parametres["transfert"]["exxkt"] = float(vals[14])

    # Conditions initiales (hardcodées en MATLAB)
    parametres.setdefault("solInitial", {})
    parametres["solInitial"]["hsini"] = 300.0
    parametres["solInitial"]["hnini"] = 300.0
    parametres["solInitial"]["hmini"] = 300.0
    parametres["solInitial"]["q0"]    = 3.06

    # Fonte CEQUEAU
    parametres.setdefault("fonte", {})
    parametres["fonte"].setdefault("cequeau", {})
    fonte = parametres["fonte"]["cequeau"]
    fonte["strne_s"] = float(vals[15])  # params(16)
    fonte["tfc_s"]   = float(vals[16])  # params(17)
    fonte["tfd_s"]   = float(vals[17])  # params(18)
    fonte["tsc_s"]   = float(vals[18])  # params(19)
    fonte["tsd_s"]   = float(vals[19])  # params(20) -> tsd_s dans ton JSON
    fonte["ttd"]     = float(vals[20])  # params(21)
    fonte["tts_s"]   = float(vals[21])  # params(22)

    # Evapo CEQUEAU
    parametres.setdefault("evapo", {})
    parametres["evapo"].setdefault("cequeau", {})
    ev = parametres["evapo"]["cequeau"]
    ev["evnap"] = float(vals[22])  # params(23)
    ev["xaa"]   = float(vals[23])  # params(24)
    ev["xit"]   = float(vals[24])  # params(25)

    return parametres

def build_surface(CEarray, CParray):
    nCE = len(CEarray)
    superficieCP = np.array([float(cp["pctSurface"]) for cp in CParray])
    superficieCE = np.zeros(nCE)
    for i, ce in enumerate(CEarray):
        CEid = ce["CEid"]
        superficieCE[i] = sum(float(cp["pctSurface"]) for cp in CParray if cp["idCE"] == CEid)
    return {
        "superficieCE": superficieCE.astype(float),
        "superficieCP": superficieCP.astype(float),
        "superficieTotale": float(superficieCP.sum()),
        "superficieBassin": float(superficieCE.sum())
    }

def load_meteo_execution_from_nc(nc_path):
    nc = Dataset(nc_path)

    pTot        = nc["pTot"][:]
    tMin        = nc["tMin"][:]
    tMax        = nc["tMax"][:]
    rayonnement = nc["rayonnement"][:]
    pression    = nc["pression"][:]
    vent        = nc["vitesseVent"][:]
    nebulosite  = nc["nebulosite"][:]

    t = (tMin + tMax) / 2.0

    meteo = {
        "tMax":        tMax.astype(np.single),
        "tMin":        tMin.astype(np.single),
        "pTot":        pTot.astype(np.single),
        "rayonnement": rayonnement.astype(np.single),
        "nebulosite":  nebulosite.astype(np.single),
        "pression":    pression.astype(np.single),
        "vitesseVent": vent.astype(np.single),
        "t":           t.astype(np.single),
    }

    def matlab_datenum(y, m, d):
        return (dt.datetime(y, m, d) - dt.datetime(1, 1, 1)).days + 366

    execution = {
        "dateDebut": float(matlab_datenum(1978, 12, 31)),
        "dateFin":   float(matlab_datenum(2025, 12, 28)),
    }

    return meteo, execution

def build_connectivity_from_bassin(base_results, musle_mat_path):
    """
    Construit:
      - routingOrder : déjà dans MUSLE_inputs.mat
      - upstreamCPs  : {1 x nCP} indices 1-based des CP amont
      - CEtoCP       : [nCE x 1] mapping CE -> CP (1-based)
    à partir de bassinVersant.json.
    """
    # 1) Charger le JSON CEQUEAU
    with open(os.path.join(base_results, "bassinVersant.json"), "r") as f:
        bv_json = json.load(f)

    ce = bv_json["carreauxEntiers"]
    cp = bv_json["carreauxPartiels"]

    nCE = len(ce["CEid"])
    nCP = len(cp["CPid"])

    # 2) Dictionnaire CPid -> index (0-based)
    cp_ids = [float(x) for x in cp["CPid"]]
    id_to_idx = {cp_ids[i]: i for i in range(nCP)}

    # 3) upstreamCPs : liste de vecteurs d'indices 1-based
    upstreamCPs = []
    for i in range(nCP):
        raw = cp["idCPsAmont"][i]  # liste ou liste de listes
        flat = []
        for x in raw:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)

        idxs = []
        for cid in flat:
            cid = float(cid)
            if cid == 0:
                continue
            j = id_to_idx.get(cid, None)
            if j is not None:
                idxs.append(j + 1)  # 1-based pour MATLAB/Octave

        upstreamCPs.append(np.array(idxs, dtype=np.int32))

    # 4) CEtoCP : pour chaque CE, on choisit un CP associé
    CEtoCP = np.zeros(nCE, dtype=np.int32)
    idCE_list = [float(x) for x in ce["CEid"]]
    idCE_cp   = [float(x) for x in cp["idCE"]]

    for i in range(nCE):
        CEid = idCE_list[i]
        # tous les CP qui appartiennent à ce CE
        candidates = [j for j in range(nCP) if idCE_cp[j] == CEid]
        if not candidates:
            CEtoCP[i] = 0  # aucun CP associé (sécurité)
        else:
            # on prend le premier (simple, robuste)
            CEtoCP[i] = candidates[0] + 1  # 1-based

    # 5) routingOrder : on le lit depuis MUSLE_inputs.mat
    musle = loadmat(musle_mat_path)
    if "routingOrder" not in musle:
        raise RuntimeError("routingOrder absent de MUSLE_inputs.mat")
    routingOrder = musle["routingOrder"].astype(np.int64).ravel()

    # 6) Sauvegarde dans un .mat pour Octave
    out_path = os.path.join(base_results, "Connectivity_for_CEQUEAU.mat")
    savemat(
        out_path,
        {
            "routingOrder": routingOrder,
            "upstreamCPs": np.array(upstreamCPs, dtype=object),
            "CEtoCP": CEtoCP,
        },
        do_compression=True,
    )

    print(f"Connectivity_for_CEQUEAU.mat créé : {out_path}")

def build_WSRStruct_light():
    base_results = r"D:\CEQUEAU-Sed\data\results"
    nc_path = r"C:\Users\jugur\Downloads\pycequeau\pycequeau\meteo\meteo_cequeau.nc"

    bassinVersant, CEarray, CParray = load_bassin_versant(base_results)
    nCE = len(CEarray)   # <--- IMPORTANT

    parametres = load_parametres(base_results)

    # Injecter tous les paramètres calibrés (sol + transfert + fonte + evapo + IC)
    parametres = apply_ruiss_parameters_from_excel(parametres)

    # === PATCH : étendre les scalaires sol en vecteurs de taille nCE ===
    def expand(v):
        if isinstance(v, (list, np.ndarray)):
            a = np.array(v, dtype=float).ravel()
            if a.size == 1:
                return np.full((nCE,), float(a[0]), dtype=float)
            return a
        else:
            return np.full((nCE,), float(v), dtype=float)

    sol = parametres["sol"]
    for key in ["cin_s", "cvnb_s", "cvnh_s", "cvsi_s",
                "hinf_s", "hint_s", "hnap_s", "hpot_s",
                "hsol_s", "hrimp_s", "tri_s"]:
        sol[key] = expand(sol[key])

    # Surface détaillée
    surface = build_surface(CEarray, CParray)
    parametres["surface"] = surface

    bassinVersant["superficieCE"] = surface["superficieCE"]
    bassinVersant["superficieCP"] = surface["superficieCP"]
    bassinVersant["superficieTotale"] = surface["superficieTotale"]
    bassinVersant["superficieBassin"] = surface["superficieBassin"]

    musle_mat_path = r"D:\SedimentPY\outputs\MUSLE_inputs.mat"
    build_connectivity_from_bassin(base_results, musle_mat_path)

    meteo, execution = load_meteo_execution_from_nc(nc_path)

    WSRStruct = {
        "bassinVersant": bassinVersant,
        "parametres": parametres,
        "meteoInterpolee": meteo,
        "execution": execution
    }

    out = r"C:\Stage\CEQUEAU\data\rebuilt_mat"
    os.makedirs(out, exist_ok=True)

    print("nCE =", nCE)
    print("cin_s shape =", np.array(parametres["sol"]["cin_s"]).shape)
    print("cvnb_s shape =", np.array(parametres["sol"]["cvnb_s"]).shape)

    savemat(os.path.join(out, "WSRStruct_light.mat"), {"WSRStruct": WSRStruct})
    savemat(os.path.join(out, "parameters.mat"),      {"parameters": parametres})
    savemat(os.path.join(out, "bassinVersant_CEQUEAU.mat"), {"bassinVersant": bassinVersant})

    print("WSRStruct_light.mat et bassinVersant_CEQUEAU.mat créés avec succès !")

if __name__ == "__main__":
    build_WSRStruct_light()
