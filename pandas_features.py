"""Pandas equivalent of spark_features.py (same features, same split logic)."""
import argparse, os, time
import pandas as pd

I32 = {"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int32", "reordered": "int8"}


def build(d, frac, shift):
    o = pd.read_csv(f"{d}/orders.csv")
    if frac < 1:
        o = o[o.user_id % 100 < int(frac * 100)]
    users = o.loc[o.eval_set == "train", "user_id"].unique()
    o = o[o.user_id.isin(users)].sort_values(["user_id", "order_number"]).copy()
    o["days_cum"] = o["days_since_prior_order"].fillna(0).groupby(o["user_id"]).cumsum()
    o["target_n"] = o.groupby("user_id")["order_number"].transform("max") - shift
    cols = ["order_id", "product_id", "add_to_cart_order", "reordered"]
    op = pd.concat([pd.read_csv(f"{d}/order_products__prior.csv", usecols=cols, dtype=I32),
                    pd.read_csv(f"{d}/order_products__train.csv", usecols=cols, dtype=I32)],
                   ignore_index=True)
    oi = o[["order_id", "user_id", "order_number", "days_cum", "target_n"]].astype(
        {"order_id": "int32", "user_id": "int32", "order_number": "int16", "target_n": "int16", "days_cum": "float32"})
    op = op.merge(oi, on="order_id")
    hist = op[op.order_number < op.target_n]
    labels = op.loc[op.order_number == op.target_n, ["user_id", "product_id"]].assign(label=1)
    del op, oi
    tgt = (o.loc[o.order_number == o.target_n,
                 ["user_id", "days_since_prior_order", "order_dow", "order_hour_of_day", "days_cum"]]
             .rename(columns={"days_since_prior_order": "t_days_since_prior", "order_dow": "t_dow",
                              "order_hour_of_day": "t_hour", "days_cum": "t_days_cum"}))
    ho = o[o.order_number < o.target_n]
    u = ho.groupby("user_id").agg(u_n_orders=("order_number", "size"),
                                  u_avg_gap=("days_since_prior_order", "mean"))
    u = u.join(hist.groupby("user_id").agg(u_total_items=("product_id", "size"),
                                           u_distinct=("product_id", "nunique"),
                                           u_reorder_ratio=("reordered", "mean")))
    u["u_avg_basket"] = u.u_total_items / u.u_n_orders
    p = hist.groupby("product_id").agg(p_orders=("order_id", "size"), p_users=("user_id", "nunique"),
                                       p_reorder_rate=("reordered", "mean"),
                                       p_avg_cart=("add_to_cart_order", "mean"))
    up = hist.groupby(["user_id", "product_id"]).agg(
        up_orders=("order_id", "size"), up_first_n=("order_number", "min"),
        up_last_n=("order_number", "max"), up_avg_cart=("add_to_cart_order", "mean"),
        up_last_days=("days_cum", "max")).reset_index()
    prod = pd.read_csv(f"{d}/products.csv", usecols=["product_id", "aisle_id", "department_id"])
    f = (up.merge(u.reset_index(), on="user_id").merge(tgt, on="user_id")
           .merge(p.reset_index(), on="product_id").merge(prod, on="product_id")
           .merge(labels, on=["user_id", "product_id"], how="left"))
    f["label"] = f["label"].fillna(0).astype("int8")
    f["up_orders_since_last"] = f.u_n_orders - f.up_last_n
    f["up_order_rate"] = f.up_orders / f.u_n_orders
    f["up_rate_since_first"] = f.up_orders / (f.u_n_orders - f.up_first_n + 1)
    f["up_days_since_last"] = f.t_days_cum - f.up_last_days
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out/pandas")
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    t0, rows = time.time(), 0
    for name, shift in [("train", 1), ("test", 0)]:
        f = build(a.data, a.frac, shift)
        rows += len(f)
        os.makedirs(f"{a.out}/{name}", exist_ok=True)
        f.to_parquet(f"{a.out}/{name}/part.parquet", index=False)
        del f
    print(f"ELAPSED_SEC={time.time() - t0:.2f} ROWS={rows}")