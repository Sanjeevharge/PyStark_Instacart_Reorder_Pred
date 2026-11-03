# PyStark_Instacart_Reorder_Pred
requires a data folder with all the csv files in the Instacart Dataset on kaggle
PS C:\Users\DELL\Downloads\files> python train.py                                                
Training until validation scores don't improve for 50 rounds
[100]   valid_0's auc: 0.836418
[200]   valid_0's auc: 0.837564
[300]   valid_0's auc: 0.838049
[400]   valid_0's auc: 0.838258
[500]   valid_0's auc: 0.838357
[600]   valid_0's auc: 0.838406
[700]   valid_0's auc: 0.838436
[800]   valid_0's auc: 0.838465
Early stopping, best iteration is:
[764]   valid_0's auc: 0.838486
test rows=8,474,661 positive rate=0.098
baseline: order_rate   AUC=0.7730  F1=0.3754  thr=0.34
baseline: in prev order F1=0.3468
LightGBM               AUC=0.8358  F1=0.4458  thr=0.22
up_rate_since_first     2.318171e+06
up_orders_since_last    2.090501e+06
up_order_rate           1.956017e+06
up_orders               1.184724e+06
up_days_since_last      5.313776e+05
p_reorder_rate          4.582023e+05
u_reorder_ratio         2.450787e+05
t_days_since_prior      1.755246e+05
p_orders                9.049685e+04
p_avg_cart              8.720499e+04
dtype: float64

frac,impl,rows,median_min
0.1,pandas,1668916,0.78
0.1,spark_naive,1668916,20.89
0.1,spark_tuned,1668916,2.5
0.5,pandas,8250663,1.01
0.5,spark_naive,8250663,22.71
0.5,spark_tuned,8250663,3.12
1.0,pandas,16387449,1.55
1.0,spark_naive,16387449,27.43
1.0,spark_tuned,16387449,3.59
