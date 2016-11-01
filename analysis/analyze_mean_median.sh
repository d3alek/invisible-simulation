cat $1 | awk '{print $1,$4,$6}' > $1.tmp
python graph_mean_median.py $1.tmp
