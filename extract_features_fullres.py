import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from PIL import Image

warnings.filterwarnings("ignore")

K = 3
DISTS = [1, 2, 3, 4, 5]
ANGS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
LVL = 256

feat_names = ["mean", "std_dev", "skewness", "kurtosis",
              "glcm_contrast", "glcm_correlation", "glcm_energy",
              "glcm_entropy", "glcm_homogeneity", "glcm_cluster_shade"]


def get_roi(grey):
    h, w = grey.shape
    px = grey.reshape(-1, 1).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3)
    lbl = km.fit_predict(px).reshape(h, w)
    means = [grey[lbl == c].mean() for c in range(K)]
    bg = int(np.argmin(means))
    cnts = {c: np.sum(lbl == c) for c in range(K) if c != bg}
    roi = max(cnts, key=cnts.get)
    return (lbl == roi).astype(np.uint8)


def gp(g, prop):
    return float(graycoprops(g, prop).sum())


def extract(path):
    try:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        print(f"  cant load {path}: {e}")
        return np.zeros(10)

    grey = (0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]).astype(np.uint8)
    mask = get_roi(grey)

    thresh = threshold_otsu(grey)
    _ = (grey >= thresh).astype(np.uint8) * mask

    mg = grey.astype(np.float64) * mask
    norm = mg / 255.0
    dn = equalize_adapthist(norm, clip_limit=0.03) * 255.0

    lo, hi = dn.min(), dn.max()
    d8 = ((dn - lo) / (hi - lo + 1e-10) * 255).astype(np.uint8)

    flat = dn[dn > 0] if (dn > 0).any() else dn.ravel()
    mn = float(flat.mean())
    sd = float(flat.std())
    sk = float(skew(flat))
    ku = float(kurtosis(flat))

    glcm = graycomatrix(d8, distances=DISTS, angles=ANGS,
                        levels=LVL, symmetric=True, normed=True)

    con = gp(glcm, "contrast")
    cor = gp(glcm, "correlation")
    eng = gp(glcm, "energy")
    hom = gp(glcm, "homogeneity")
    eps = 1e-10
    ent = float(-np.sum(glcm * np.log2(glcm + eps)))

    p_n = glcm / (glcm.sum(axis=(0,1), keepdims=True) + eps)
    idx = np.arange(LVL)
    mu_i = (idx.reshape(-1,1,1,1) * p_n).sum(axis=(0,1))
    mu_j = (idx.reshape(1,-1,1,1) * p_n).sum(axis=(0,1))
    cs = float(((idx.reshape(-1,1,1,1) + idx.reshape(1,-1,1,1) - mu_i - mu_j)**3 * p_n).sum())

    return np.array([mn, sd, sk, ku, con, cor, eng, ent, hom, cs], dtype=np.float64)


def do_row(row, data_dir):
    folder = row["source_folder"]
    fname = row["original_filename"]
    p = os.path.join(data_dir, folder, fname)
    if not os.path.exists(p):
        p = os.path.join(data_dir, folder, fname + ".tif")
    if not os.path.exists(p):
        print(f" not found: {p}")
        return None

    f = extract(p)
    r = dict(zip(feat_names, f))
    r["maoe_risk"] = int(row["maoe_risk"])
    r["maoe_susceptibility"] = int(row["maoe_risk"] > 0)
    r["source_folder"] = folder
    r["original_filename"] = fname
    r["angle"] = int(row["angle"])
    return r


def run(meta, data_dir, label, njobs):
    print(f"  extracting {len(meta)} images ({label}) ...")
    results = Parallel(n_jobs=njobs, verbose=5)(
        delayed(do_row)(row, data_dir) for _, row in meta.iterrows()
    )
    rows = [r for r in results if r is not None]
    print(f"  got {len(rows)}/{len(meta)}")
    cols = feat_names + ["maoe_risk", "maoe_susceptibility", "source_folder", "original_filename", "angle"]
    return pd.DataFrame(rows, columns=cols)


def main():
   

    meta_tr = pd.read_csv("meta_train.csv")
    meta_te = pd.read_csv("meta_test.csv")
    print(f"train: {len(meta_tr)}   test: {len(meta_te)}")

    df_train = run(meta_tr, '.', "train", -1)
    df_train.to_csv("features_train.csv", index=False)
    print(f"saved features_train.csv ")

    df_test = run(meta_te, '.', "test", -1)
    df_test.to_csv("features_test.csv", index=False)
    print(f"saved features_test.csv ")

    print("done")


if __name__ == "__main__":
    main()
