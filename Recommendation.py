# %% [markdown]
# !pip -q install --upgrade numpy pandas scipy
# !pip -q install implicit==0.7.2

# -*- coding: utf-8 -*-
# ============================================================
# ALS (implicit) + BM25 + recency + cooc-rerank — QUICK mode
# + EASE^R backend (normalizat) + BLEND (EASE+ALS) + COOC backend
# + Cold-start pseudo-embedding (brand/categorie/subcategorie)
# + Fallback inteligent pe brand/categorie/subcategorie + popularitate
# - Leave-last-per-user; evaluare on-the-fly
# - NU salvează CSV/JSON (doar artefact .pkl cu modelul final)
# - Artefact final: /content/drive/MyDrive/produc_best.pkl
# ============================================================

import os, glob, json as _json, math, pickle, random, warnings, time, hashlib, traceback
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")
import pandas as pd
from datetime import datetime, timezone
from collections import Counter, defaultdict
from itertools import combinations, product
import scipy.sparse as sp

import implicit
from implicit.nearest_neighbours import bm25_weight

# ---------------- Master config ----------------
ORDERS_PATH = None
USE_ONLY_COMPLETED = True
SEED = 42
TEST_SPLIT = 0.20
MIN_BASKET_LEN = 2
MIN_USER_EVENTS = 3
MIN_ITEM_EVENTS = 1

# Evaluare
EVAL_TOPKS = (5, 10)
PRIMARY_K = 5

# *** QUICK vs FULL ***
QUICK_MODE = True  # <--- pune False pentru tunning mare

# Sweep control
FULL_SWEEP  = not QUICK_MODE

# Griduri — QUICK (rapid) vs FULL (extins)
if QUICK_MODE:
    HALF_LIFE_GRID = [120.0, 180.0]     # ajustat după sezonalitatea dataset-ului tău
    BM25_GRID = [(1.2, 0.75)]
    ALS_GRID = [
        {"K":192, "IT":40, "REG":0.10, "ALPHA":55.0},
        {"K":256, "IT":45, "REG":0.11, "ALPHA":60.0},
    ]
    RERANK_PROFILES = [
        dict(TAU=1.00, ALPHA=0.75, BETA=0.14, GAMMA=0.08,  DELTA=0.02,
             POP_EPS=-0.006, LAMBDA_SHRINK=38.0, CAND_POOL=200,
             MMR_ENABLE=True,  MMR_LAMBDA=0.95, MMR_MIN_RATIO=2.0,
             RERANK_ONLY_LONGTAIL=True,  POP_PCTL=60, POOL_BASE=40, POOL_EXTRA=100),
    ]
    SIM_GRID = [
        {"sim_backend": "ALS"},
        {"sim_backend": "EASE",  "ease_lambda": 300.0},
        {"sim_backend": "BLEND", "ease_lambda": 300.0, "w_ease": 0.30},
        {"sim_backend": "COOC"},
    ]
else:
    HALF_LIFE_GRID = [30.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    BM25_GRID = [(1.0, 0.65), (1.2, 0.75), (1.4, 0.75), (1.6, 0.75)]
    ALS_GRID = [
        {"K":192, "IT":40, "REG":0.09, "ALPHA":55.0},
        {"K":256, "IT":45, "REG":0.11, "ALPHA":60.0},
        {"K":320, "IT":45, "REG":0.12, "ALPHA":65.0},
        {"K":384, "IT":50, "REG":0.13, "ALPHA":70.0},
        {"K":448, "IT":55, "REG":0.14, "ALPHA":75.0},
    ]
    RERANK_PROFILES = [
        dict(TAU=1.00, ALPHA=0.75, BETA=0.14, GAMMA=0.08,  DELTA=0.02,
             POP_EPS=-0.006, LAMBDA_SHRINK=38.0, CAND_POOL=220,
             MMR_ENABLE=True,  MMR_LAMBDA=0.94, MMR_MIN_RATIO=2.0,
             RERANK_ONLY_LONGTAIL=True,  POP_PCTL=60, POOL_BASE=45, POOL_EXTRA=100),
        dict(TAU=1.05, ALPHA=0.73, BETA=0.18, GAMMA=0.06,  DELTA=0.03,
             POP_EPS=-0.004, LAMBDA_SHRINK=40.0, CAND_POOL=260,
             MMR_ENABLE=True,  MMR_LAMBDA=0.95, MMR_MIN_RATIO=2.0,
             RERANK_ONLY_LONGTAIL=True,  POP_PCTL=65, POOL_BASE=50, POOL_EXTRA=120),
    ]
    EASE_LAMBDA_GRID = [75.0, 150.0, 300.0, 600.0, 1000.0]
    BLEND_WEIGHTS    = [0.2, 0.4, 0.6]
    SIM_GRID = (
        [{"sim_backend": "ALS"}] +
        [{"sim_backend": "EASE",  "ease_lambda": lam} for lam in EASE_LAMBDA_GRID] +
        [{"sim_backend": "BLEND", "ease_lambda": lam, "w_ease": w} for lam in EASE_LAMBDA_GRID for w in BLEND_WEIGHTS] +
        [{"sim_backend": "COOC"}]
    )

TOP_NEIGHBORS = 40
NUM_THREADS = max(1, os.cpu_count() or 4)
USE_GPU = bool(int(os.getenv("ALS_GPU", "0")))  # export ALS_GPU=1 dacă ai CUDA

# Artefact
BEST_PKL = "/content/drive/MyDrive/produc_best.pkl"

# ---- Praguri "Bought Together" (soft only, fără filtre hard) ----
BT_MIN_SUPPORT = 10
BT_MIN_CONF    = 0.08
BT_MIN_LIFT    = 1.05

# *** Reguli runtime: NU filtra strict pe BT ***
RECOMMEND_ONLY_BT = False
BT_EVAL_STRICT    = False

# ---- FALLBACK pe brand/categorie/subcategorie ----
FALLBACK_ENABLE       = True
FB_W_COS              = 0.70
FB_W_POP              = 0.30
FB_BRAND_BONUS        = 0.12
FB_CATEGORY_BONUS     = 0.08
FB_SUBCAT_BONUS       = 0.00
FB_MAX_CAND_BRAND     = 5000
FB_MAX_CAND_CATEGORY  = 5000
FB_MAX_CAND_SUBCAT    = 0

# Bonus long-tail în rescoring
LT_BONUS = 0.03

random.seed(SEED); np.random.seed(SEED)

# ---------------- IO helpers ----------------
def find_orders_path():
    cands = sorted(glob.glob("*.json"), key=os.path.getmtime, reverse=True)
    for p in cands:
        if "orders" in p.lower():
            return p
    if os.path.exists("orders_generated_xl.json"):
        return "orders_generated_xl.json"
    raise FileNotFoundError("Nu găsesc fișierul orders_*.json. Încarcă un JSON cu comenzi.")

if ORDERS_PATH is None:
    ORDERS_PATH = find_orders_path()
print(f"[INFO] Folosesc comenzi din: {ORDERS_PATH}")

def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return _json.load(f)
        except _json.JSONDecodeError as e:
            f.seek(0)
            raw = f.read()
            cutoff = raw[:e.pos]
            last_valid = max(cutoff.rfind('}'), cutoff.rfind(']'))
            if last_valid != -1:
                try:
                    return _json.loads(f"[{cutoff[:last_valid+1]}]")
                except Exception:
                    return []
            return []

orders = safe_load_json(ORDERS_PATH)
print(f"[INFO] Loaded orders: {len(orders)}")

# ---------------- Split ----------------
def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

def user_temporal_split(orders, test_ratio=0.2):
    by_user = {}
    for o in orders:
        u = o.get("clientId")
        dt = parse_dt(o.get("orderDate"))
        if u is None or dt is None:
            continue
        if USE_ONLY_COMPLETED and o.get("status") != "COMPLETED":
            continue
        by_user.setdefault(u, []).append(o)
    train, test = [], []
    for _, lst in by_user.items():
        lst.sort(key=lambda x: parse_dt(x["orderDate"]))
        cut = max(1, int(len(lst) * (1 - test_ratio)))
        train += lst[:cut]
        test += lst[cut:]
    train.sort(key=lambda x: parse_dt(x["orderDate"]))
    test.sort(key=lambda x: parse_dt(x["orderDate"]))
    return train, test

train_orders, test_orders = user_temporal_split(orders, TEST_SPLIT)
print(f"[INFO] Train: {len(train_orders)} | Test: {len(test_orders)} (leave-last per user)")

# ---------------- Extrage META (brand/categorie/subcategorie) ----------------
BRAND_TOKENS    = ["brand", "brandname", "manufacturer", "manufacturername", "marca", "producator", "producer", "vendor", "supplier", "fabricant"]
CATEGORY_TOKENS = ["category", "categoryname", "categorie", "productcategory", "dept", "department"]
SUBCAT_TOKENS   = ["subcategory", "subcategorie", "sub_category", "subcat", "subdept", "subdepartment", "family", "subfamily", "line", "group"]

def _is_primitive(x):
    return isinstance(x, (str, int, float)) and str(x).strip().lower() not in ("", "none", "null", "nan")

def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    yield k2, v2
                    yield from _walk(v2)

def _extract_token_values(op, tokens):
    found = []
    for k, v in _walk(op):
        kl = str(k).lower()
        if any(t in kl for t in tokens):
            if _is_primitive(v):
                found.append(str(v))
            elif isinstance(v, dict):
                for name_key in ("name", "title", "label", "value"):
                    if name_key in v and _is_primitive(v[name_key]):
                        found.append(str(v[name_key]))
        if isinstance(v, dict) and ("name" in v and "value" in v):
            n = str(v.get("name", "")).lower()
            if any(t in n for t in tokens) and _is_primitive(v.get("value")):
                found.append(str(v["value"]))
    return [s.strip() for s in found if _is_primitive(s)]

def _norm_key(s: str) -> str:
    return " ".join(str(s).strip().split()).casefold()

def build_item_metadata(orders_all):
    tmp = {}
    for o in orders_all:
        for op in (o.get("orderProducts", []) or []):
            pid = op.get("productId")
            if pid is None:
                continue
            pid = int(pid)
            rec = tmp.setdefault(pid, {
                "brand": Counter(), "category": Counter(), "subcategory": Counter(),
                "brand_alias": {}, "category_alias": {}, "subcategory_alias": {}
            })
            # direct fields
            if _is_primitive(op.get("brandName")):
                b = str(op["brandName"]).strip()
                bk = _norm_key(b)
                rec["brand"][bk] += 1; rec["brand_alias"].setdefault(bk, b)
            if _is_primitive(op.get("category")):
                c = str(op["category"]).strip()
                ck = _norm_key(c)
                rec["category"][ck] += 1; rec["category_alias"].setdefault(ck, c)
            if _is_primitive(op.get("subcategory")):
                sc = str(op["subcategory"]).strip()
                sck = _norm_key(sc)
                rec["subcategory"][sck] += 1; rec["subcategory_alias"].setdefault(sck, sc)
            # deep tokens
            for b in _extract_token_values(op, BRAND_TOKENS):
                bk = _norm_key(b); rec["brand"][bk] += 1; rec["brand_alias"].setdefault(bk, b)
            for c in _extract_token_values(op, CATEGORY_TOKENS):
                ck = _norm_key(c); rec["category"][ck] += 1; rec["category_alias"].setdefault(ck, c)
            for sc in _extract_token_values(op, SUBCAT_TOKENS):
                sck = _norm_key(sc); rec["subcategory"][sck] += 1; rec["subcategory_alias"].setdefault(sck, sc)

    item_meta = {}
    brand_index = defaultdict(set)
    category_index = defaultdict(set)
    subcat_index = defaultdict(set)

    for pid, cnts in tmp.items():
        if cnts["brand"]:
            bk, _ = cnts["brand"].most_common(1)[0]
            b_display = cnts["brand_alias"].get(bk, bk)
        else:
            bk, b_display = None, None
        if cnts["category"]:
            ck, _ = cnts["category"].most_common(1)[0]
            c_display = cnts["category_alias"].get(ck, ck)
        else:
            ck, c_display = None, None
        if cnts["subcategory"]:
            sck, _ = cnts["subcategory"].most_common(1)[0]
            sc_display = cnts["subcategory_alias"].get(sck, sck)
        else:
            sck, sc_display = None, None

        item_meta[pid] = {
            "brand": b_display, "brand_key": bk,
            "category": c_display, "category_key": ck,
            "subcategory": sc_display, "subcategory_key": sck
        }
        if bk is not None: brand_index[bk].add(pid)
        if ck is not None: category_index[ck].add(pid)
        if sck is not None: subcat_index[sck].add(pid)

    brand_index = {k: sorted(list(v)) for k, v in brand_index.items()}
    category_index = {k: sorted(list(v)) for k, v in category_index.items()}
    subcat_index = {k: sorted(list(v)) for k, v in subcat_index.items()}

    print(f"[META] produse cu brand: {sum(1 for m in item_meta.values() if m['brand'] is not None)} | categorie: {sum(1 for m in item_meta.values() if m['category'] is not None)} | subcategorie: {sum(1 for m in item_meta.values() if m['subcategory'] is not None)}")
    return item_meta, brand_index, category_index, subcat_index

ITEM_META_ALL, BRAND_INDEX_ALL, CATEGORY_INDEX_ALL, SUBCAT_INDEX_ALL = build_item_metadata(orders)

# --- (nou) Nume produse pentru printurile BT (opțional, dacă există în JSON) ---
def build_item_names(orders_all):
    names = {}
    name_keys = ["productName", "name", "title", "product", "skuName", "displayName"]
    for o in orders_all:
        for op in (o.get("orderProducts", []) or []):
            pid = op.get("productId")
            if pid is None:
                continue
            pid = int(pid)
            if pid in names:
                continue
            for k in name_keys:
                v = op.get(k)
                if isinstance(v, (str, int, float)) and str(v).strip():
                    names[pid] = str(v).strip()
                    break
    return names

ITEM_NAMES = build_item_names(orders)

# ---------------- Co-oc pe TRAIN ----------------
item_count, pair_count = Counter(), Counter()
train_baskets = []
for o in train_orders:
    s = {op.get("productId") for op in (o.get("orderProducts", []) or []) if op.get("productId") is not None}
    if len(s) >= 2:
        ss = set(map(int, s))
        train_baskets.append(ss)
        for i in ss:
            item_count[i] += 1
        for a, b in combinations(sorted(ss), 2):
            pair_count[(a, b)] += 1
N_train_baskets = max(1, len(train_baskets))
MAX_ITEM_COUNT = max(item_count.values()) if item_count else 1

def percentile_pop(pctl: int):
    return float(np.percentile(list(item_count.values()), pctl)) if item_count else 0.0

# Parametri dinamici (setați per-profil în buclă)
RERANK_ONLY_LONGTAIL = True
POP_THRESHOLD = percentile_pop(60)
ALPHA = 0.75
BETA = 0.14
GAMMA = 0.08
DELTA = 0.02
TAU = 1.0
POP_EPS = -0.006
LAMBDA_SHRINK = 38.0
CAND_POOL = 200
MMR_ENABLE = True
MMR_LAMBDA = 0.95
MMR_MIN_RATIO = 2.0
POOL_BASE = 40
POOL_EXTRA = 90

# ---- Co-ocerență (pentru rescoring & BT) ----
def cooc_stats(a, b):
    if a == b:
        return 0, 0, 0, 0.0, 0.0, 0.0
    x, y = (a, b) if a < b else (b, a)
    supp = pair_count.get((x, y), 0)
    ca, cb = item_count.get(a, 0), item_count.get(b, 0)
    if supp == 0 or ca == 0 or cb == 0:
        return supp, ca, cb, 0.0, 0.0, 0.0
    pa, pb, pab = ca / N_train_baskets, cb / N_train_baskets, supp / N_train_baskets
    lift = pab / (pa * pb + 1e-12)
    lift_c = min(lift, 5.0) / 5.0
    PMI = math.log((pab + 1e-12) / (pa * pb + 1e-12))
    NPMI = PMI / (-math.log(pab + 1e-12))
    npmi01 = max(0.0, min(1.0, 0.5 * (NPMI + 1.0)))
    jacc = supp / (ca + cb - supp + 1e-12)
    shrink = supp / (supp + LAMBDA_SHRINK)
    return supp, ca, cb, lift_c * shrink, npmi01 * shrink, jacc * shrink

def conf_bonus(supp):
    return 1.0 / (1.0 + math.exp(-(supp - 1.5)))  # mai blând ca înainte

def rescoring(anchor_pid, cand_pid, base_sim):
    base_sim = math.tanh(float(base_sim) / max(1e-9, TAU))
    supp, ca, cb, lift_c, npmi01, jacc = cooc_stats(anchor_pid, cand_pid)
    # NU mai anulăm semnalele la suport mic — shrink face treaba
    base  = ALPHA*base_sim + BETA*lift_c + GAMMA*npmi01 + DELTA*jacc
    score = 0.96*base + 0.04*conf_bonus(supp)

    if MAX_ITEM_COUNT > 0:
        pop_norm = math.log1p(item_count.get(cand_pid, 0)) / math.log1p(MAX_ITEM_COUNT)
        score += POP_EPS * pop_norm

    if item_count.get(int(cand_pid), 0) < POP_THRESHOLD:
        score += LT_BONUS
    return score

def _adaptive_pool_size(anchor_pid, base=POOL_BASE, extra=POOL_EXTRA):
    pop = item_count.get(anchor_pid, 0)
    pop_norm = math.log1p(pop) / math.log1p(MAX_ITEM_COUNT) if MAX_ITEM_COUNT > 0 else 0.0
    return int(max(10, min(CAND_POOL, base + (1.0 - pop_norm) * extra)))

def _is_longtail(pid):
    return item_count.get(int(pid), 0) < POP_THRESHOLD if RERANK_ONLY_LONGTAIL else True

# ---------------- Build interactions ----------------
def build_train_df(train_orders, half_life_days: float):
    now = datetime.now(timezone.utc)
    rows = []
    user_event_count = Counter()
    for o in train_orders:
        u = o.get("clientId")
        if u is None:
            continue
        user_event_count[u] += len([op for op in (o.get("orderProducts", []) or []) if op.get("productId") is not None])
    for o in train_orders:
        u = o.get("clientId")
        if u is None:
            continue
        odt = parse_dt(o.get("orderDate")) or now
        if odt.tzinfo is None:
            odt = odt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - odt).days)
        adj = 0.75 if user_event_count[u] > 30 else (1.25 if user_event_count[u] < 5 else 1.0)
        hl = half_life_days * adj
        w_time = math.exp(-math.log(2.0) * (age_days / hl))
        seen = set()
        for op in (o.get("orderProducts", []) or []):
            pid = op.get("productId")
            if pid is None:
                continue
            pid = int(pid)
            if pid in seen:
                continue
            seen.add(pid)
            qty = float(op.get("quantity", 1))
            w_qty = np.log1p(qty)
            rows.append((u, pid, float(w_qty * w_time)))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["user", "item", "weight"])
    if MIN_USER_EVENTS > 1:
        uc = df.groupby("user")["item"].count()
        df = df[df["user"].isin(uc[uc >= MIN_USER_EVENTS].index)]
    if MIN_ITEM_EVENTS > 1:
        ic = df.groupby("item")["user"].count()
        df = df[df["item"].isin(ic[ic >= MIN_ITEM_EVENTS].index)]
    return df if len(df) > 0 else None

def df_to_users_items(df):
    users = np.sort(df["user"].unique())
    items = np.sort(df["item"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    p2j = {p: j for j, p in enumerate(items)}
    ui = df["user"].map(u2i).values
    ij = df["item"].map(p2j).values
    wt = df["weight"].astype(np.float32).values
    R_counts = sp.csr_matrix((wt, (ui, ij)), shape=(len(users), len(items)))
    return R_counts, users, items

# caching pentru HL/BM25
_CACHE_RC  = {}  # hl -> (R_counts, users_arr, items_arr)
_CACHE_RIU = {}  # (id(R_counts_T), K1, B) -> bm25(items x users)

def make_R_counts(half_life_days: float):
    if half_life_days in _CACHE_RC:
        return _CACHE_RC[half_life_days]
    df = build_train_df(train_orders, half_life_days)
    assert df is not None and len(df) > 0, "Nu există interacțiuni valide după filtrare."
    R_counts, users_arr, items_arr = df_to_users_items(df)
    print(f"[HL={half_life_days}] Users: {len(users_arr)} | Items: {len(items_arr)} | NNZ: {R_counts.nnz}")
    _CACHE_RC[half_life_days] = (R_counts, users_arr, items_arr)
    return _CACHE_RC[half_life_days]

def get_Riu_from_Rcounts(R_counts_T, K1, B):
    key = (id(R_counts_T), K1, B)
    if key in _CACHE_RIU:
        return _CACHE_RIU[key]
    Riu = bm25_weight(R_counts_T, K1=K1, B=B).tocsr().astype(np.float32)
    _CACHE_RIU[key] = Riu
    return Riu

def train_als_get_Vn_from_Riu(Riu, k, iters, reg, alpha, items_arr):
    Riu = Riu.tocsr(copy=True)
    Riu.data = 1.0 + float(alpha) * Riu.data   # Hu et al.
    als = implicit.als.AlternatingLeastSquares(
        factors=int(k), iterations=int(iters), regularization=float(reg),
        random_state=SEED, calculate_training_loss=False,
        use_gpu=USE_GPU, num_threads=NUM_THREADS
    )
    als.fit(Riu)  # (items x users)
    V = als.item_factors if als.item_factors.shape[0] == len(items_arr) else als.user_factors
    V = V.astype(np.float32)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return Vn

# ---- Normalizare/Autoscalare similitudini matriceale ----
def _normalize_ease(S: np.ndarray, mode="row", clip_neg=True) -> np.ndarray:
    if clip_neg:
        S = np.maximum(S, 0.0)
    if mode == "row":
        denom = np.sqrt((S * S).sum(axis=1, keepdims=True)) + 1e-12
        S = S / denom
    elif mode == "sym":
        d = np.clip(S.sum(axis=1), 1e-12, None)
        dinv = 1.0 / np.sqrt(d)
        S = (S * dinv[:, None]) * dinv[None, :]
    np.fill_diagonal(S, 0.0)
    return S.astype(np.float32, copy=False)

# ---- EASE^R: cache & training ----
_CACHE_EASE = {}  # key: (id(Riu_base), ease_lambda) -> S_ease (items x items)

def train_ease_get_S_from_Riu(Riu_items_users: sp.csr_matrix, ease_lambda: float) -> np.ndarray:
    key = (id(Riu_items_users), float(ease_lambda))
    if key in _CACHE_EASE:
        return _CACHE_EASE[key]
    G = (Riu_items_users @ Riu_items_users.T).toarray().astype(np.float64)
    n = G.shape[0]
    G.flat[::n+1] += ease_lambda  # + λI
    P = np.linalg.inv(G)
    B = -P / (np.diag(P)[:, None] + 1e-12)
    np.fill_diagonal(B, 0.0)
    S = 0.5 * (B + B.T)
    S = _normalize_ease(S, mode="row", clip_neg=True)
    _CACHE_EASE[key] = S
    return S

# ---- COOC backend (lift/npmi/jaccard) ----
_CACHE_Cooc = {}
def train_cooc_get_S(items_index_local: np.ndarray) -> np.ndarray:
    key = tuple(map(int, items_index_local.tolist()))
    if key in _CACHE_Cooc: return _CACHE_Cooc[key]
    n = len(items_index_local)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = int(items_index_local[i])
        for j in range(i+1, n):
            b = int(items_index_local[j])
            supp, ca, cb, lift_c, npmi01, jacc = cooc_stats(a, b)
            sim = 0.65 * lift_c + 0.25 * npmi01 + 0.10 * jacc
            if sim > 0:
                S[i, j] = sim; S[j, i] = sim
    S = _normalize_ease(S, mode="row", clip_neg=True)
    _CACHE_Cooc[key] = S
    return S

# ---- Sim provider (setat per-trial în main) ----
_SIM_BACKEND = "ALS"
_EASE_S_ACTIVE = None   # np.ndarray [items x items] sau None
_BLEND_EASE_W = 0.30    # când BLEND, ponderea EASE; restul merge la ALS

def _autoscale_backend_sims(sims: np.ndarray) -> np.ndarray:
    q = np.quantile(np.abs(sims), 0.95) + 1e-12
    sims = sims / q
    return sims

def _sims_row(ii: int, Vn: np.ndarray, EASE_S: np.ndarray) -> np.ndarray:
    if _SIM_BACKEND == "ALS":
        sims = Vn @ Vn[ii]
    elif _SIM_BACKEND == "EASE":
        sims = _autoscale_backend_sims(EASE_S[ii].copy())
    elif _SIM_BACKEND == "BLEND":
        sims_als  = Vn @ Vn[ii]
        sims_ease = _autoscale_backend_sims(EASE_S[ii])
        sims = float(_BLEND_EASE_W) * sims_ease + (1.0 - float(_BLEND_EASE_W)) * sims_als
    elif _SIM_BACKEND == "COOC":
        sims = _autoscale_backend_sims(EASE_S[ii].copy())
    else:
        sims = Vn @ Vn[ii]
    sims[ii] = -1.0
    return sims.astype(np.float32, copy=False)

# ---------------- Test baskets ----------------
def build_test_baskets(orders_subset):
    baskets = []
    for o in orders_subset:
        s = {op.get("productId") for op in (o.get("orderProducts", []) or []) if op.get("productId") is not None}
        if len(s) >= MIN_BASKET_LEN:
            baskets.append(set(map(int, s)))
    return baskets

test_baskets_all = build_test_baskets(test_orders)
print(f"[INFO] Test baskets (raw): {len(test_baskets_all)}")

# ---------------- Helpers "Bought Together" (soft) ----------------
def _pair_measures(a: int, b: int):
    if a == b:
        return 0, 0.0, 0.0
    x, y = (a, b) if a < b else (b, a)
    supp = int(pair_count.get((x, y), 0))
    ca, cb = int(item_count.get(a, 0)), int(item_count.get(b, 0))
    if supp == 0 or ca == 0 or cb == 0 or N_train_baskets == 0:
        return 0, 0.0, 0.0
    pab = supp / N_train_baskets
    pa  = ca   / N_train_baskets
    pb  = cb   / N_train_baskets
    conf_ab = supp / ca if ca else 0.0
    lift    = pab / (pa * pb + 1e-12) if (pa > 0 and pb > 0) else 0.0
    return int(supp), float(conf_ab), float(lift)

def _pm3(a: int, b: int):
    s, c, l = _pair_measures(a, b)
    return int(s), float(c), float(l)

def _is_bt_pair(a: int, b: int):
    supp, conf_ab, lift = _pm3(a, b)
    return (supp >= BT_MIN_SUPPORT) and (conf_ab >= BT_MIN_CONF) and (lift > BT_MIN_LIFT)

def _bt_filter_for_anchor(anchor_pid: int, chosen_js, items_index_local):
    if not RECOMMEND_ONLY_BT:
        return list(chosen_js)
    kept = []
    for j in chosen_js:
        cand_pid = int(items_index_local[int(j)])
        if _is_bt_pair(int(anchor_pid), cand_pid):
            kept.append(int(j))
    return kept

# --- (nou) Print rezumat reguli A→B cu confidence ca metrică de regulă (fără CSV) ---
def print_bt_summary(top_n: int = 20):
    """
    Afișează la stdout top reguli A→B ordonate după confidence (P(B|A)),
    respectând pragurile BT_MIN_SUPPORT / BT_MIN_CONF / BT_MIN_LIFT.
    NOTĂ: 'confidence' este METRICĂ DE REGULĂ, NU acuratețea sistemului.
    """
    if not pair_count:
        print("[BT] pair_count este gol; nu există coșuri/train suficient.")
        return

    rules = []
    N = float(N_train_baskets)
    for (a, b), supp in pair_count.items():
        ca = item_count.get(a, 0); cb = item_count.get(b, 0)
        if ca == 0 or cb == 0:
            continue
        pab = supp / N
        pa  = ca   / N
        pb  = cb   / N
        lift = pab / (pa * pb + 1e-12) if (pa > 0 and pb > 0) else 0.0

        # A->B
        conf_ab = supp / ca if ca else 0.0
        if supp >= BT_MIN_SUPPORT and conf_ab >= BT_MIN_CONF and lift > BT_MIN_LIFT:
            rules.append({
                "A": int(a), "B": int(b),
                "conf": float(conf_ab),
                "support": int(supp),
                "lift": float(lift),
                "count_A": int(ca), "count_B": int(cb)
            })
        # B->A
        conf_ba = supp / cb if cb else 0.0
        if supp >= BT_MIN_SUPPORT and conf_ba >= BT_MIN_CONF and lift > BT_MIN_LIFT:
            rules.append({
                "A": int(b), "B": int(a),
                "conf": float(conf_ba),
                "support": int(supp),
                "lift": float(lift),
                "count_A": int(cb), "count_B": int(ca)
            })

    if not rules:
        print(f"[BT] Nicio regulă nu trece pragurile (support>={BT_MIN_SUPPORT}, conf>={BT_MIN_CONF}, lift>{BT_MIN_LIFT}).")
        return

    rules.sort(key=lambda r: (r["conf"], r["support"], r["lift"]), reverse=True)
    top_n = max(1, int(top_n))
    top_rules = rules[:top_n]

    # Header explicativ
    print("\n[BT] Rezumat reguli A→B (confidence = P(B|A)) — METRICĂ DE REGULĂ, nu acuratețe de sistem.")
    print(f"[BT] Praguri: support>={BT_MIN_SUPPORT}, conf>={BT_MIN_CONF}, lift>{BT_MIN_LIFT}")
    print(f"[BT] Afișez top {len(top_rules)} reguli după confidence:")

    def _nm(pid):
        nm = ITEM_NAMES.get(int(pid), "")
        return (nm if len(nm) <= 60 else (nm[:57] + "..."))

    for i, r in enumerate(top_rules, start=1):
        a, b = r["A"], r["B"]
        print(f"{i:>3}. {a} → {b} | conf={r['conf']*100:6.2f}% | support={r['support']:>4} | lift={r['lift']:.3f} | "
              f"count(A)={r['count_A']:>4} | count(B)={r['count_B']:>4} | "
              f"A_name='{_nm(a)}' | B_name='{_nm(b)}'")

    # Small recap
    best = top_rules[0]
    print(f"[BT] Max confidence observat A→B: {best['conf']*100:.2f}% (support={best['support']}, lift={best['lift']:.3f})")

# ---------------- Evaluare (on-the-fly) ----------------
def _make_eval_baskets_for_items(items_index_local):
    item_index_local = {int(pid): int(i) for i, pid in enumerate(items_index_local)}
    eval_b = []
    for b in test_baskets_all:
        bb = {pid for pid in b if pid in item_index_local}
        if len(bb) >= MIN_BASKET_LEN:
            eval_b.append(bb)
    return eval_b, item_index_local

def _mmr_select_on_pool(Vn, indices_pool, scores_pool, k, lambda_mmr=0.95):
    if not MMR_ENABLE or k <= 1 or len(indices_pool) == 0:
        order = np.argsort(scores_pool)[::-1]
        return list(indices_pool[order[:max(0, k)]].astype(int))
    V_pool = Vn[indices_pool]
    try:
        S_pool = V_pool @ V_pool.T
    except MemoryError:
        order = np.argsort(scores_pool)[::-1]
        return list(indices_pool[order[:max(0, k)]].astype(int))
    selected_pos = []
    cand_mask = np.ones(len(indices_pool), dtype=bool)
    for _ in range(min(k, len(indices_pool))):
        if not selected_pos:
            j = int(np.argmax(scores_pool)); selected_pos.append(j); cand_mask[j] = False
            continue
        max_sim_sel = np.max(S_pool[:, selected_pos], axis=1)
        mmr_scores = lambda_mmr * scores_pool - (1.0 - lambda_mmr) * max_sim_sel
        mmr_scores[~cand_mask] = -1e9
        j = int(np.argmax(mmr_scores))
        if mmr_scores[j] <= -1e8:
            break
        selected_pos.append(j); cand_mask[j] = False
    return list(indices_pool[np.array(selected_pos, dtype=int)].astype(int))

def _pick_with_rerank(ii, Vn, items_index_local, sims_row, K):
    n_items = sims_row.shape[0]
    kk = max(1, min(K, n_items - 1))
    m = _adaptive_pool_size(int(items_index_local[ii]), base=POOL_BASE, extra=POOL_EXTRA)
    m = max(kk, min(m, n_items - 1))
    pool = np.argpartition(sims_row, -m)[-m:]
    pool = pool[np.argsort(sims_row[pool])[::-1]]
    anchor_pid = int(items_index_local[ii])
    if not _is_longtail(anchor_pid):
        return list(pool[:kk].astype(int))
    rescored_scores = np.asarray([
        rescoring(anchor_pid, int(items_index_local[int(j)]), float(sims_row[int(j)]))
        for j in pool
    ], dtype=np.float32)
    if MMR_ENABLE and (len(pool) >= int(MMR_MIN_RATIO * kk)):
        return _mmr_select_on_pool(Vn, pool, rescored_scores, kk, lambda_mmr=MMR_LAMBDA)
    else:
        order = np.argsort(rescored_scores)[::-1]
        return list(pool[order[:kk]].astype(int))

def _eval_for_K(Vn, K, items_index_local, eval_baskets_local, item_index_local, use_rerank=True):
    total = hits = 0
    precisions, recalls, ap_list, rr_list = [], [], [], []
    rec_glob = set()
    n_items = Vn.shape[0]
    if n_items <= 1:
        return {"cases": 0, "HitRate@K": 0, "Precision@K": 0, "Recall@K": 0, "MAP@K": 0, "MRR@K": 0, "Coverage@K": 0}
    kk_global = max(1, min(K, n_items - 1))
    for basket in eval_baskets_local:
        for anchor in basket:
            targets = set(basket) - {anchor}
            if not targets:
                continue
            total += 1
            ii = item_index_local.get(int(anchor))
            if ii is None:
                continue
            sims_row = _sims_row(ii, Vn, _EASE_S_ACTIVE)
            if use_rerank:
                chosen_js = _pick_with_rerank(ii, Vn, items_index_local, sims_row, kk_global)
            else:
                top = np.argpartition(sims_row, -kk_global)[-kk_global:]
                chosen_js = top[np.argsort(sims_row[top])[::-1]]
            if BT_EVAL_STRICT:
                chosen_js = _bt_filter_for_anchor(int(items_index_local[ii]), chosen_js, items_index_local)
            rec_ids = [int(items_index_local[int(j)]) for j in chosen_js]
            rec_glob.update(rec_ids)
            inter = targets.intersection(rec_ids)
            if inter:
                hits += 1
            precisions.append(len(inter) / kk_global)
            recalls.append(len(inter) / len(targets))
            cum_rel = 0.0; sum_prec = 0.0
            for rank, pid in enumerate(rec_ids, start=1):
                if pid in targets:
                    cum_rel += 1; sum_prec += cum_rel / rank
            denom = min(len(targets), kk_global)
            ap_list.append(sum_prec / denom if denom > 0 else 0.0)
            rr = 0.0
            for rank, pid in enumerate(rec_ids, start=1):
                if pid in targets:
                    rr = 1.0 / rank; break
            rr_list.append(rr)
    cov = len(rec_glob) / len(items_index_local) if len(items_index_local) else 0.0
    return {
        "cases": int(total),
        "HitRate@K": float(hits / total) if total else 0.0,
        "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "MAP@K": float(np.mean(ap_list)) if ap_list else 0.0,
        "MRR@K": float(np.mean(rr_list)) if rr_list else 0.0,
        "Coverage@K": float(cov),
    }

def evaluate_model(Vn, items_index_local, eval_baskets_local, item_index_local):
    out = {}
    for K in EVAL_TOPKS:
        base = _eval_for_K(Vn, K, items_index_local, eval_baskets_local, item_index_local, use_rerank=False)
        rer  = _eval_for_K(Vn, K, items_index_local, eval_baskets_local, item_index_local, use_rerank=True)
        out[f"BASE K={K}"]   = base
        out[f"RERANK K={K}"] = rer
    return out

def pick_key(metrics_dict, primary_k=PRIMARY_K):
    m = metrics_dict.get(f"RERANK K={primary_k}", {})
    return (m.get("MAP@K", 0.0), m.get("Precision@K", 0.0), m.get("Recall@K", 0.0))

# ---------------- Cold-start & Fallback (brand/cat/subcat) ----------------
def _pop_norm_from_counts(cnt_dict, pid, max_cnt):
    cnt = cnt_dict.get(int(pid), 0); mx = max(1, int(max_cnt))
    return math.log1p(cnt) / math.log1p(mx)

def _recommend_fallback_bcs(model, product_id, k=5):
    """Fallback smart: brand/categorie/subcategorie + cos(optional) + popularitate"""
    cfg = model.get("fallback_cfg", {})
    if not cfg.get("enable", True):
        return []
    pid = int(product_id)
    item_meta = model.get("item_meta", {})
    meta = item_meta.get(pid, {})
    brand_key = meta.get("brand_key"); cat_key = meta.get("category_key"); sub_key = meta.get("subcategory_key")

    brand_index = model.get("brand_index", {}) or {}
    cat_index   = model.get("category_index", {}) or {}
    sub_index   = model.get("subcategory_index", {}) or {}
    item_index  = model.get("item_index", {})
    Vn = model.get("norm_item_factors", None)
    has_vec = (Vn is not None) and (pid in item_index)

    w_cos = float(cfg.get("w_cos", FB_W_COS))
    w_pop = float(cfg.get("w_pop", FB_W_POP))
    b_bonus = float(cfg.get("brand_bonus", FB_BRAND_BONUS))
    c_bonus = float(cfg.get("category_bonus", FB_CATEGORY_BONUS))
    s_bonus = float(cfg.get("subcategory_bonus", FB_SUBCAT_BONUS))

    cand_pids = set()
    if brand_key and brand_key in brand_index:
        for p in brand_index[brand_key][:FB_MAX_CAND_BRAND]:
            if p != pid:
                cand_pids.add(int(p))
    if cat_key and cat_key in cat_index:
        for p in cat_index[cat_key][:FB_MAX_CAND_CATEGORY]:
            if p != pid:
                cand_pids.add(int(p))
    if sub_key and sub_key in sub_index:
        for p in sub_index[sub_key][:FB_MAX_CAND_SUBCAT]:
            if p != pid:
                cand_pids.add(int(p))

    if not cand_pids:
        tops = model.get("top_popular_fallback", [])[:k]
        return [(int(p), 0.0) for p in tops]

    cand_pids = list(cand_pids)
    scores = []
    if has_vec:
        ii = item_index[pid]
        present = [p for p in cand_pids if p in item_index]
        if present:
            cand_idx = np.array([item_index[p] for p in present], dtype=np.int64)
            sims = (Vn @ Vn[ii])[cand_idx].astype(np.float32)
            sims[ii] = -1.0
            for p, c in zip(present, sims):
                s = w_cos * float(c) + w_pop * _pop_norm_from_counts(model.get("item_count", {}), p, model.get("max_item_count", 1))
                pm = item_meta.get(p, {})
                if brand_key and pm.get("brand_key") == brand_key: s += b_bonus
                if cat_key   and pm.get("category_key") == cat_key: s += c_bonus
                if sub_key   and pm.get("subcategory_key") == sub_key: s += s_bonus
                scores.append((int(p), float(s)))
        absent = [p for p in cand_pids if p not in present]
        for p in absent:
            s = w_pop * _pop_norm_from_counts(model.get("item_count", {}), p, model.get("max_item_count", 1))
            pm = item_meta.get(p, {})
            if brand_key and pm.get("brand_key") == brand_key: s += b_bonus
            if cat_key   and pm.get("category_key") == cat_key: s += c_bonus
            if sub_key   and pm.get("subcategory_key") == sub_key: s += s_bonus
            scores.append((int(p), float(s)))
    else:
        for p in cand_pids:
            s = w_pop * _pop_norm_from_counts(model.get("item_count", {}), p, model.get("max_item_count", 1))
            pm = item_meta.get(p, {})
            if brand_key and pm.get("brand_key") == brand_key: s += b_bonus
            if cat_key   and pm.get("category_key") == cat_key: s += c_bonus
            if sub_key   and pm.get("subcategory_key") == sub_key: s += s_bonus
            scores.append((int(p), float(s)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def _coldstart_pseudo_embedding_for_pid(pid, k, Vn, item_index_map, item_count_local,
                                        item_meta_all, brand_index_all, category_index_all, subcat_index_all,
                                        w_brand=0.15, w_cat=0.35, w_sub=0.50, max_cand=5000):
    """Centroid semantic (brand/categorie/subcategorie) + scorare cosine+popularitate+bonusuri"""
    pid = int(pid)
    meta = item_meta_all.get(pid, {})
    bk = meta.get("brand_key"); ck = meta.get("category_key"); sk = meta.get("subcategory_key")
    if Vn is None or (not bk and not ck and not sk):
        return []

    # candidați
    cand_pids = set()
    if bk and bk in brand_index_all: cand_pids.update(brand_index_all[bk][:max_cand])
    if ck and ck in category_index_all: cand_pids.update(category_index_all[ck][:max_cand])
    if sk and sk in subcat_index_all: cand_pids.update(subcat_index_all[sk][:max_cand])
    cand_pids.discard(pid)
    cand_pids = [p for p in cand_pids if p in item_index_map]
    if not cand_pids:
        return []

    # centroid
    d = Vn.shape[1]
    v = np.zeros(d, dtype=np.float32)
    if bk and bk in brand_index_all:
        idx_b = [item_index_map[p] for p in brand_index_all[bk] if p in item_index_map]
        if idx_b: v += w_brand * Vn[np.array(idx_b, dtype=np.int64)].mean(axis=0)
    if ck and ck in category_index_all:
        idx_c = [item_index_map[p] for p in category_index_all[ck] if p in item_index_map]
        if idx_c: v += w_cat * Vn[np.array(idx_c, dtype=np.int64)].mean(axis=0)
    if sk and sk in subcat_index_all:
        idx_s = [item_index_map[p] for p in subcat_index_all[sk] if p in item_index_map]
        if idx_s: v += w_sub * Vn[np.array(idx_s, dtype=np.int64)].mean(axis=0)
    nrm = np.linalg.norm(v)
    if nrm == 0: return []
    v = v / nrm

    cand_idx = np.array([item_index_map[p] for p in cand_pids], dtype=np.int64)
    sims = (Vn[cand_idx] @ v).astype(np.float32)

    # scorare finală
    out = []
    mx_cnt = max(1, max(item_count_local.values()) if item_count_local else 1)
    for p, s in zip(cand_pids, sims):
        pm = item_meta_all.get(p, {})
        bb = FB_BRAND_BONUS if (bk and pm.get("brand_key") == bk) else 0.0
        cc = FB_CATEGORY_BONUS if (ck and pm.get("category_key") == ck) else 0.0
        ss = FB_SUBCAT_BONUS if (sk and pm.get("subcategory_key") == sk) else 0.0
        pop_norm = _pop_norm_from_counts(item_count_local, p, mx_cnt)
        score = 0.70*float(s) + 0.30*(1.0 - pop_norm) + bb + cc + ss
        out.append((int(p), float(score)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]

# ---------------- Precompute vecini pt best ----------------
def precompute_neighbors_in_model(Vn, items_index_local):
    n_items = Vn.shape[0]
    precomp = {}
    if n_items <= 1:
        return precomp
    for ii in range(n_items):
        anchor_pid = int(items_index_local[ii])
        sims_row = _sims_row(ii, Vn, _EASE_S_ACTIVE)
        kk = TOP_NEIGHBORS
        chosen_js = _pick_with_rerank(ii, Vn, items_index_local, sims_row, kk)
        out = []
        for j in chosen_js:
            pid = int(items_index_local[int(j)])
            base_sim = float(sims_row[int(j)])
            sc = rescoring(anchor_pid, pid, base_sim) if _is_longtail(anchor_pid) else base_sim
            out.append((pid, float(np.float16(sc))))
        precomp[int(anchor_pid)] = out
    return precomp

def precompute_neighbors_all(Vn, items_index_local):
    """Precompute pentru TOATE produsele din orders: în model -> rerank sim; în afara modelului -> cold-start centroid."""
    inmodel_pre = precompute_neighbors_in_model(Vn, items_index_local)
    items_index_set = set(map(int, items_index_local))
    all_pids = set(map(int, ITEM_META_ALL.keys()))
    out = dict(inmodel_pre)
    item_index_map = {int(pid): int(i) for i, pid in enumerate(items_index_local)}
    for pid in (all_pids - items_index_set):
        recs = _coldstart_pseudo_embedding_for_pid(
            pid, TOP_NEIGHBORS, Vn, item_index_map, item_count,
            ITEM_META_ALL, BRAND_INDEX_ALL, CATEGORY_INDEX_ALL, SUBCAT_INDEX_ALL
        )
        if not recs:
            # fallback popular + bonusuri semantice
            meta = ITEM_META_ALL.get(pid, {})
            bk, ck, sk = meta.get("brand_key"), meta.get("category_key"), meta.get("subcategory_key")
            cand = set()
            if bk and bk in BRAND_INDEX_ALL: cand.update(BRAND_INDEX_ALL[bk][:FB_MAX_CAND_BRAND])
            if ck and ck in CATEGORY_INDEX_ALL: cand.update(CATEGORY_INDEX_ALL[ck][:FB_MAX_CAND_CATEGORY])
            if sk and sk in SUBCAT_INDEX_ALL: cand.update(SUBCAT_INDEX_ALL[sk][:FB_MAX_CAND_SUBCAT])
            cand.discard(pid)
            scores = []
            for p in cand:
                s = 0.30 * _pop_norm_from_counts(item_count, p, MAX_ITEM_COUNT)
                pm = ITEM_META_ALL.get(p, {})
                if bk and pm.get("brand_key") == bk: s += FB_BRAND_BONUS
                if ck and pm.get("category_key") == ck: s += FB_CATEGORY_BONUS
                if sk and pm.get("subcategory_key") == sk: s += FB_SUBCAT_BONUS
                scores.append((int(p), float(s)))
            if not scores:
                tops = [int(pp) for pp, _ in sorted(item_count.items(), key=lambda x: x[1], reverse=True)[:TOP_NEIGHBORS]]
                recs = [(int(p), 0.0) for p in tops if int(p) != int(pid)]
            else:
                scores.sort(key=lambda x: x[1], reverse=True)
                recs = scores[:TOP_NEIGHBORS]
        out[int(pid)] = recs
    return out

# ---------------- Persist helpers ----------------
def save_best_artifact(Vn, items_index_local, params, precomp, item_count_local, best_pkl=BEST_PKL):
    TOP_POPULAR = [int(pid) for pid, _ in sorted(item_count_local.items(), key=lambda x: x[1], reverse=True)[:TOP_NEIGHBORS]]

    items_set = set(map(int, items_index_local))
    item_meta_reduced = {int(pid): meta for pid, meta in ITEM_META_ALL.items()}  # toate produsele

    def _reduce_index(full_index, max_keep):
        return {k: sorted(v)[:max_keep] for k, v in full_index.items()}

    brand_index_red    = _reduce_index(BRAND_INDEX_ALL,    FB_MAX_CAND_BRAND)
    category_index_red = _reduce_index(CATEGORY_INDEX_ALL, FB_MAX_CAND_CATEGORY)
    subcat_index_red   = _reduce_index(SUBCAT_INDEX_ALL,   FB_MAX_CAND_SUBCAT)

    artifact = {
        "item_index": {int(pid): int(i) for i, pid in enumerate(items_index_local)},
        "index_item": {int(i): int(pid) for i, pid in enumerate(items_index_local)},
        "items_index": items_index_local.tolist(),
        "norm_item_factors": Vn.astype(np.float16),
        "params": params,
        "precomputed_neighbors_reranked": {"top": int(TOP_NEIGHBORS), "neighbors": precomp},
        "top_popular_fallback": TOP_POPULAR,
        "bt_only": bool(RECOMMEND_ONLY_BT),
        "bt_thresholds": {"min_support": BT_MIN_SUPPORT, "min_conf": BT_MIN_CONF, "min_lift": BT_MIN_LIFT},
        "item_meta": item_meta_reduced,
        "brand_index": brand_index_red,
        "category_index": category_index_red,
        "subcategory_index": subcat_index_red,
        "item_count": {int(pid): int(cnt) for pid, cnt in item_count_local.items()},
        "max_item_count": int(max(item_count_local.values()) if item_count_local else 1),
        "fallback_cfg": dict(enable=FALLBACK_ENABLE, w_cos=FB_W_COS, w_pop=FB_W_POP,
                             brand_bonus=FB_BRAND_BONUS, category_bonus=FB_CATEGORY_BONUS, subcategory_bonus=FB_SUBCAT_BONUS),
        # pentru reproducere backend best
        "best_sim_backend": params.get("sim_backend", "ALS"),
        "best_blend_w_ease": params.get("blend_w_ease", 0.0),
    }
    os.makedirs(os.path.dirname(best_pkl), exist_ok=True)
    with open(best_pkl, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Best model salvat: {best_pkl}")

# ---------------- Main sweep ----------------
def main():
    global RERANK_ONLY_LONGTAIL, POP_THRESHOLD
    global ALPHA, BETA, GAMMA, DELTA, TAU, POP_EPS, LAMBDA_SHRINK, CAND_POOL
    global MMR_ENABLE, MMR_LAMBDA, MMR_MIN_RATIO, POOL_BASE, POOL_EXTRA
    global _SIM_BACKEND, _EASE_S_ACTIVE, _BLEND_EASE_W

    best = None
    combos = list(product(
        HALF_LIFE_GRID,
        BM25_GRID,
        range(len(ALS_GRID)),
        range(len(RERANK_PROFILES)),
        range(len(SIM_GRID))               # include backendul
    ))
    total_combos = len(combos)
    print(f"[INFO] Combinații totale: {total_combos}")

    t0 = time.time()

    for t_idx, (hl, (K1, B), als_idx, rr_idx, sim_idx) in enumerate(combos, start=1):
        sim_cfg = SIM_GRID[sim_idx]
        sim_backend = sim_cfg["sim_backend"]
        ease_lambda = float(sim_cfg.get("ease_lambda", 300.0))
        blend_w_ease = float(sim_cfg.get("w_ease", 0.30))

        print(f"\n===== Trial {t_idx}/{total_combos} | HL={hl} | BM25(K1={K1}, B={B}) | "
              f"ALS={ALS_GRID[als_idx]} | RERANK_PROFILE#{rr_idx} | SIM={sim_backend}"
              f"{' (λ='+str(ease_lambda)+')' if sim_backend in ('EASE','BLEND') else ''} =====")

        # profil rerank
        prof = RERANK_PROFILES[rr_idx]
        RERANK_ONLY_LONGTAIL = bool(prof["RERANK_ONLY_LONGTAIL"])
        POP_THRESHOLD = percentile_pop(int(prof["POP_PCTL"]))
        ALPHA = prof["ALPHA"]; BETA = prof["BETA"]; GAMMA = prof["GAMMA"]; DELTA = prof["DELTA"]
        TAU = prof["TAU"]; POP_EPS = prof["POP_EPS"]; LAMBDA_SHRINK = prof["LAMBDA_SHRINK"]; CAND_POOL = prof["CAND_POOL"]
        MMR_ENABLE = bool(prof["MMR_ENABLE"]); MMR_LAMBDA = prof["MMR_LAMBDA"]; MMR_MIN_RATIO = prof["MMR_MIN_RATIO"]
        POOL_BASE = prof["POOL_BASE"]; POOL_EXTRA = prof["POOL_EXTRA"]

        # backend de similitudine
        _SIM_BACKEND   = sim_backend
        _BLEND_EASE_W  = blend_w_ease
        _EASE_S_ACTIVE = None

        try:
            # 1) Build + BM25 (cache)
            R_counts, users_arr, items_arr = make_R_counts(hl)
            R_counts_T = R_counts.T
            Riu_base = get_Riu_from_Rcounts(R_counts_T, K1, B)  # items x users

            # 2) ALS
            cfg = ALS_GRID[als_idx]
            Vn_try = train_als_get_Vn_from_Riu(
                Riu=Riu_base, k=cfg["K"], iters=cfg["IT"], reg=cfg["REG"],
                alpha=cfg["ALPHA"], items_arr=items_arr
            )

            # 2.1) EASE/COOC — doar dacă e cazul
            items_index_local = np.asarray(items_arr, dtype=np.int64)
            if sim_backend == "EASE" or sim_backend == "BLEND":
                _EASE_S_ACTIVE = train_ease_get_S_from_Riu(Riu_base, ease_lambda=ease_lambda)
            elif sim_backend == "COOC":
                _EASE_S_ACTIVE = train_cooc_get_S(items_index_local)
            else:
                _EASE_S_ACTIVE = None

            eval_baskets_local, item_index_local = _make_eval_baskets_for_items(items_index_local)

            # 3) Evaluate
            metrics = evaluate_model(Vn_try, items_index_local, eval_baskets_local, item_index_local)
            key = pick_key(metrics, primary_k=PRIMARY_K)
            print(f"[Trial score @K={PRIMARY_K}] MAP={key[0]:.6f} | Prec={key[1]:.6f} | Rec={key[2]:.6f}")

            if (best is None) or (key > best["key"]):
                best = {"key": key, "params": {
                            "seed": SEED, "use_only_completed": USE_ONLY_COMPLETED,
                            "split": "leave-last-per-user", "test_split": TEST_SPLIT,
                            "min_user_events": MIN_USER_EVENTS, "min_item_events": MIN_ITEM_EVENTS,
                            "half_life": hl, "bm25_K1": K1, "bm25_B": B, **cfg,
                            "rerank_profile": rr_idx,
                            "rerank_alpha": ALPHA, "rerank_beta": BETA, "rerank_gamma": GAMMA, "rerank_delta": DELTA,
                            "tau": TAU, "pop_eps": POP_EPS, "lambda_shrink": LAMBDA_SHRINK, "cand_pool": CAND_POOL,
                            "mmr_enable": MMR_ENABLE, "mmr_lambda": MMR_LAMBDA, "mmr_min_ratio": MMR_MIN_RATIO,
                            "pool_base": POOL_BASE, "pool_extra": POOL_EXTRA,
                            "rerank_only_longtail": RERANK_ONLY_LONGTAIL, "pop_pctl": prof["POP_PCTL"],
                            # ---- backend:
                            "sim_backend": sim_backend, "ease_lambda": ease_lambda, "blend_w_ease": blend_w_ease,
                        },
                        "Vn": Vn_try, "items_index": items_index_local,
                        "ease_S": (_EASE_S_ACTIVE.copy() if _EASE_S_ACTIVE is not None else None),
                        "sim_backend": sim_backend, "blend_w_ease": blend_w_ease}
                print(f"[BEST↑] Nou best @K={PRIMARY_K}: MAP={key[0]:.6f} Prec={key[1]:.6f} Rec={key[2]:.6f}")

        except AssertionError as e:
            print(f"[WARN] Trial sărit: {e}")
        except Exception as e:
            print(f"[ERROR] Trial failure: {type(e).__name__}: {e}")
            traceback.print_exc()

    assert best is not None, "Sweep nu a produs niciun setup valid."

    # 4) Setează profilul best + backend pt precompute final
    prof_idx = best["params"]["rerank_profile"]
    prof = RERANK_PROFILES[prof_idx]
    globals().update({
        "RERANK_ONLY_LONGTAIL": bool(prof["RERANK_ONLY_LONGTAIL"]),
        "POP_THRESHOLD": percentile_pop(int(prof["POP_PCTL"])),
        "ALPHA": prof["ALPHA"], "BETA": prof["BETA"], "GAMMA": prof["GAMMA"], "DELTA": prof["DELTA"],
        "TAU": prof["TAU"], "POP_EPS": prof["POP_EPS"], "LAMBDA_SHRINK": prof["LAMBDA_SHRINK"],
        "CAND_POOL": prof["CAND_POOL"], "MMR_ENABLE": bool(prof["MMR_ENABLE"]),
        "MMR_LAMBDA": prof["MMR_LAMBDA"], "MMR_MIN_RATIO": prof["MMR_MIN_RATIO"],
        "POOL_BASE": prof["POOL_BASE"], "POOL_EXTRA": prof["POOL_EXTRA"],
    })
    # backend
    _SIM_BACKEND = best["sim_backend"]; _BLEND_EASE_W = best["blend_w_ease"]
    _EASE_S_ACTIVE = best["ease_S"]

    # 5) Precompute vecini pentru TOATE produsele (in/out model)
    print("\n[PRECOMPUTE] Calculez vecinii pentru toate produsele (in-model + cold-start)…")
    precomp_all = precompute_neighbors_all(best["Vn"], best["items_index"])

    # 6) Salvează best artifact
    save_best_artifact(best["Vn"], best["items_index"], best["params"], precomp_all, item_count, best_pkl=BEST_PKL)

    dt = time.time() - t0
    sc = best["key"]
    print(f"\n[FINISH] Best @K={PRIMARY_K}: MAP={sc[0]:.6f} Prec={sc[1]:.6f} Rec={sc[2]:.6f}")
    print(f"[FINISH] Artefact .pkl final: {BEST_PKL}")
    print(f"[DONE] Durată totală: {dt/60.0:.2f} min")

    try:
        print_bt_summary(top_n=20)  # ajustează top_n dacă vrei mai multe linii în print
    except Exception as e:
        print(f"[BT] Eroare la sumarul BT: {e}")

# ---------------- Inference helper ----------------
def load_model(pkl_path=BEST_PKL):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def recommend_for_product(model, product_id, k=5):
    pid = int(product_id)
    pre = model.get("precomputed_neighbors_reranked", {}).get("neighbors", {})
    if pid in pre and pre[pid]:
        return pre[pid][:k]

    # cold-start: pseudo-embedding din centroid semantic
    item_index = model.get("item_index", {})
    Vn = model.get("norm_item_factors", None)
    if (pid not in item_index) and (Vn is not None):
        # reconstruim harta minimă necesară
        item_index_map = item_index
        item_count_local = model.get("item_count", {})
        recs = _coldstart_pseudo_embedding_for_pid(
            pid, k, Vn.astype(np.float32), item_index_map, item_count_local,
            model.get("item_meta", {}), model.get("brand_index", {}), model.get("category_index", {}), model.get("subcategory_index", {})
        )
        if recs:
            return recs

    # fallback brand/cat/subcat + popularitate
    fb = _recommend_fallback_bcs(model, pid, k=k)
    if fb:
        return fb

    # fallback final: top popular
    tops = model.get("top_popular_fallback", [])[:k]
    return [(int(p), 0.0) for p in tops]

# ---------------- Run ----------------
if __name__ == "__main__":
    main()
