import os, sys, glob, warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROT_ANGLES = [30, 60, 90, 120]
IMG_SZ = (256, 256)
SEED = 42
TEST_SZ = 0.25

img_cols = ["Image name", "image name", "Filename", "filename", "Name"]
risk_cols = ["Risk of macular edema", "risk of macular edema",
             "Retinal risk of macular edema", "Macular edema risk"]


def find_col(df, opts):
    for c in opts:
        if c in df.columns:
            return c
    raise KeyError("column not found")


def read_ann(path):
    ext = os.path.splitext(path)[1].lower()
    eng = "xlrd" if ext == ".xls" else "openpyxl"
    df = pd.read_excel(path, engine=eng)
    df.columns = df.columns.str.strip()
    ic = find_col(df, img_cols)
    mc = find_col(df, risk_cols)
    out = pd.DataFrame({
        "filename": df[ic].astype(str).str.strip(),
        "maoe_risk": df[mc].astype(int),
    })
    out["maoe_susceptibility"] = (out["maoe_risk"] > 0).astype(int)
    return out


def rot(arr, angle):
    return np.array(Image.fromarray(arr).rotate(angle, resample=Image.BICUBIC, expand=False))


def load_img(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SZ, Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"could not load {path}: {e}")
        return None


def load_messidor(root):
    images, labels, meta = [], [], []

    base_dirs = sorted(glob.glob(os.path.join(root, "Base*")))
    if not base_dirs:
        raise FileNotFoundError(f"no Base* folders in '{root}'")

    for bd in base_dirs:
        folder = os.path.basename(bd)
        excels = glob.glob(os.path.join(bd, "*.xlsx")) + glob.glob(os.path.join(bd, "*.xls"))
        if not excels:
            print(f"no excel in {folder}, skipping")
            continue

        try:
            ann = read_ann(excels[0])
        except:
            continue

        for _, row in ann.iterrows():
            fname = row["filename"]
            p = os.path.join(bd, fname)
            if not os.path.exists(p):
                p = os.path.join(bd, fname + ".tif")
            if not os.path.exists(p):
                continue

            orig = load_img(p)
            if orig is None:
                continue

            risk = int(row["maoe_risk"])
            susc = int(row["maoe_susceptibility"])

            images.append(orig)
            labels.append(risk)
            meta.append({"source_folder": folder, "original_filename": fname,
                         "angle": 0, "maoe_risk": risk, "maoe_susceptibility": susc})

            for a in ROT_ANGLES:
                images.append(rot(orig, a))
                labels.append(risk)
                meta.append({"source_folder": folder, "original_filename": fname,
                             "angle": a, "maoe_risk": risk, "maoe_susceptibility": susc})
        print("loaded : ", bd)

    if not images:
        raise RuntimeError("no images loaded")

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    mdf = pd.DataFrame(meta)

    uniq = mdf["original_filename"].unique()
    ulabels = (mdf.drop_duplicates("original_filename")
                   .set_index("original_filename")["maoe_risk"]
                   .loc[uniq].values)

    tr_imgs, te_imgs = train_test_split(
        uniq, test_size=TEST_SZ, stratify=ulabels, random_state=SEED)

    tr_mask = mdf["original_filename"].isin(tr_imgs).values
    te_mask = mdf["original_filename"].isin(te_imgs).values

    Xtr, Xte = X[tr_mask], X[te_mask]
    ytr, yte = y[tr_mask], y[te_mask]
    mtr = mdf[tr_mask].reset_index(drop=True)
    mte = mdf[te_mask].reset_index(drop=True)

    return Xtr, Xte, ytr, yte, mtr, mte


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, mtr, mte = load_messidor('.')
    print(f"train: {Xtr.shape}  test: {Xte.shape}")
    mtr.to_csv("meta_train.csv", index=False)
    mte.to_csv("meta_test.csv", index=False)
    print("saved meta csv")