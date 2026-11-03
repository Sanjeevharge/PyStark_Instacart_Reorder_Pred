"""Spark feature pipeline. Usage:
python spark_features.py --data data --out out/full --frac 1.0 --mode tuned
Temporal design: for each user, test target = last order (eval_set=train),
train target = second-to-last order. Features use only orders BEFORE the target.
"""
import argparse, os, time
from pyspark.sql import SparkSession, functions as F, Window
import os
import sys

# Set Hadoop home directory for Windows
os.environ['HADOOP_HOME'] = "C:/hadoop"
# Tell winutils where to find the bin folder
sys.path.append("C:/hadoop/bin")
OSCH = ("order_id INT, user_id INT, eval_set STRING, order_number INT, order_dow INT, "
        "order_hour_of_day INT, days_since_prior_order DOUBLE")
PSCH = "order_id INT, product_id INT, add_to_cart_order INT, reordered INT"
PRSCH = "product_id INT, product_name STRING, aisle_id INT, department_id INT"
CORES = os.cpu_count() or 4


def session(mode):
    b = (SparkSession.builder.appName(f"feat-{mode}").master("local[*]")
         .config("spark.driver.memory", os.environ.get("SPARK_MEM", "8g")))
    if mode == "naive":   # defaults-style: 200 shuffle parts, no AQE, no broadcast, no cache
        b = (b.config("spark.sql.shuffle.partitions", "200")
              .config("spark.sql.adaptive.enabled", "false")
              .config("spark.sql.autoBroadcastJoinThreshold", "-1"))
    else:                 # tuned: partitions ~ 2x cores, AQE on
        b = (b.config("spark.sql.shuffle.partitions", str(CORES * 2))
              .config("spark.sql.adaptive.enabled", "true"))
    s = b.getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    return s


def build(spark, d, frac, shift, tuned):
    npart = CORES * 2
    rd = lambda f, sch: spark.read.csv(f"{d}/{f}", header=True, schema=sch, escape='"')
    o = rd("orders.csv", OSCH)
    if frac < 1:
        o = o.filter(F.col("user_id") % 100 < int(frac * 100))
    users = o.filter(F.col("eval_set") == "train").select("user_id").distinct()
    if tuned:
        users = F.broadcast(users)
    o = o.join(users, "user_id", "left_semi")
    w = Window.partitionBy("user_id").orderBy("order_number")
    o = (o.withColumn("days_cum", F.sum(F.coalesce("days_since_prior_order", F.lit(0.0))).over(w))
          .withColumn("target_n", F.max("order_number").over(Window.partitionBy("user_id")) - shift))
    if tuned:
        o = o.repartition(npart, "user_id").cache()

    op = (rd("order_products__prior.csv", PSCH).unionByName(rd("order_products__train.csv", PSCH))
          .join(o.select("order_id", "user_id", "order_number", "days_cum", "target_n"), "order_id"))
    if tuned:
        op = op.repartition(npart, "user_id").cache()   # reused by hist + labels

    hist = op.filter(F.col("order_number") < F.col("target_n"))
    labels = (op.filter(F.col("order_number") == F.col("target_n"))
                .select("user_id", "product_id").withColumn("label", F.lit(1)))
    tgt = (o.filter(F.col("order_number") == F.col("target_n"))
            .select("user_id",
                    F.col("days_since_prior_order").alias("t_days_since_prior"),
                    F.col("order_dow").alias("t_dow"),
                    F.col("order_hour_of_day").alias("t_hour"),
                    F.col("days_cum").alias("t_days_cum")))
    ho = o.filter(F.col("order_number") < F.col("target_n"))

    u1 = ho.groupBy("user_id").agg(F.count("*").alias("u_n_orders"),
                                   F.avg("days_since_prior_order").alias("u_avg_gap"))
    u2 = hist.groupBy("user_id").agg(F.count("*").alias("u_total_items"),
                                     F.countDistinct("product_id").alias("u_distinct"),
                                     F.avg("reordered").alias("u_reorder_ratio"))
    u = (u1.join(u2, "user_id")
           .withColumn("u_avg_basket", F.col("u_total_items") / F.col("u_n_orders")))
    p = hist.groupBy("product_id").agg(F.count("*").alias("p_orders"),
                                       F.countDistinct("user_id").alias("p_users"),
                                       F.avg("reordered").alias("p_reorder_rate"),
                                       F.avg("add_to_cart_order").alias("p_avg_cart"))
    up = hist.groupBy("user_id", "product_id").agg(
        F.count("*").alias("up_orders"), F.min("order_number").alias("up_first_n"),
        F.max("order_number").alias("up_last_n"), F.avg("add_to_cart_order").alias("up_avg_cart"),
        F.max("days_cum").alias("up_last_days"))
    prod = rd("products.csv", PRSCH).select("product_id", "aisle_id", "department_id")
    if tuned:
        p, prod = F.broadcast(p), F.broadcast(prod)

    return (up.join(u, "user_id").join(tgt, "user_id").join(p, "product_id").join(prod, "product_id")
              .join(labels, ["user_id", "product_id"], "left")
              .withColumn("label", F.coalesce("label", F.lit(0)))
              .withColumn("up_orders_since_last", F.col("u_n_orders") - F.col("up_last_n"))
              .withColumn("up_order_rate", F.col("up_orders") / F.col("u_n_orders"))
              .withColumn("up_rate_since_first",
                          F.col("up_orders") / (F.col("u_n_orders") - F.col("up_first_n") + 1))
              .withColumn("up_days_since_last", F.col("t_days_cum") - F.col("up_last_days")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out/full")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--mode", choices=["naive", "tuned"], default="tuned")
    a = ap.parse_args()
    spark = session(a.mode)
    t0 = time.time()
    for name, shift in [("train", 1), ("test", 0)]:
        build(spark, a.data, a.frac, shift, a.mode == "tuned").write.mode("overwrite").parquet(f"{a.out}/{name}")
        spark.catalog.clearCache()
    el = time.time() - t0
    rows = sum(spark.read.parquet(f"{a.out}/{n}").count() for n in ("train", "test"))
    print(f"ELAPSED_SEC={el:.2f} ROWS={rows}")
