set -e
python spark_features.py --data data --out out/full --frac 1.0 --mode tuned
python train.py
python bench.py
