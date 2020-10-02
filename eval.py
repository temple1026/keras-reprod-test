import numpy as np
import os
from datetime import datetime

def getResults(path_result, n_splits, num_train):
    with open(path_result, "r") as f:
        lines = f.readlines()

    results = []
    dates = []

    for line in lines:
        l = line.splitlines()[0].split()

        if len(l) != 1:
            results.append(l)
        else:
            dates.append(datetime.strptime(l[0], "%Y%m%d-%H%M%S"))

    results = np.array(results).astype(np.float64)
    results = np.reshape(results, newshape=(num_train, n_splits, 3))
    
    var = np.mean(np.var(results[:,:,1:], axis=0))
    diff_time_total = np.sum([dates[i + 1] - dates[i] for i in range(0, num_train-1)])

    print(f"{path_result}: {var:.4}, {diff_time_total}")
    return

def main():
    print("Variance, Time_diff_total")
    getResults(path_result=os.path.join("results", "with_tf_option.txt"), n_splits=10, num_train=10)
    getResults(path_result=os.path.join("results", "without_tf_option.txt"), n_splits=10, num_train=10)
    return

if __name__=="__main__":
    main()