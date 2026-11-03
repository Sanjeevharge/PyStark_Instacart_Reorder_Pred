"""LightGBM on Spark-built features. Train: 2nd-to-last order labels. Test: each user's last order."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score

def load(p):
    df = pd.read_parquet(p)
    for c in df.columns:
        if df[c].dtype == "float64": df[c] = df[c].astype("float32")
    return df
tr = load("out/full/train"); te = load("out/full/test")
feats = [c for c in tr.columns if c not in ("user_id", "product_id", "label")]
va_mask = tr.user_id % 10 == 0          # user-level validation split (early stopping + threshold)
va, tr = tr[va_mask], tr[~va_mask]


def best_thr(y, s):
    ts = np.arange(0.1, 0.61, 0.02)
    return ts[int(np.argmax([f1_score(y, s >= t) for t in ts]))]


def report(name, s_va, s_te):
    t = best_thr(va.label, s_va)
    print(f"{name:<22} AUC={roc_auc_score(te.label, s_te):.4f}  F1={f1_score(te.label, s_te >= t):.4f}  thr={t:.2f}")


m = lgb.train(dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
                   min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, verbose=-1),
              lgb.Dataset(tr[feats], tr.label), 1000, valid_sets=[lgb.Dataset(va[feats], va.label)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
print(f"test rows={len(te):,} positive rate={te.label.mean():.3f}")
report("baseline: order_rate", va.up_order_rate, te.up_order_rate)
rule_f1 = f1_score(te.label, te.up_orders_since_last == 0)
print(f"{'baseline: in prev order':<22} F1={rule_f1:.4f}")
report("LightGBM", m.predict(va[feats]), m.predict(te[feats]))
print(pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False).head(10))