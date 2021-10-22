import pandas as pd
import numpy as np


def generate_dataframes(all_results):
    column_names = []
    means_results = []
    ratios_results = []
    approx_ratios_results = []

    for item in all_results.items():
        name, res = item
        column_names.append(name)

        m = [np.mean(l) for l in res[0]]
        means_results.append(m.copy())

        m = [np.mean(l) for l in res[1]]
        ratios_results.append(m.copy())

        m = [np.mean(l) for l in res[2]]
        approx_ratios_results.append(m.copy())

    means_results = np.stack(means_results, axis=1)
    ratios_results = np.stack(ratios_results, axis=1)
    approx_ratios_results = np.stack(approx_ratios_results, axis=1)
    
    means_df = pd.DataFrame(means_results, columns=column_names)
    # start index = p at 1
    means_df.index = np.arange( 1, len(means_df) + 1)
    means_df.index.name = "p"
    # round 
    means_df = means_df.round(2)
    
    ratio_df = pd.DataFrame(ratios_results, columns=column_names)
    # start index = p at 1
    ratio_df.index = np.arange( 1, len(ratio_df) + 1)
    ratio_df.index.name = "p"
    # round 
    ratio_df = ratio_df.round(2)
    
    approx_ratios_df = pd.DataFrame(approx_ratios_results, columns=column_names)
    # start index = p at 1
    approx_ratios_df.index = np.arange( 1, len(approx_ratios_df) + 1)
    approx_ratios_df.index.name = "p"
    # round 
    approx_ratios_df = approx_ratios_df.round(2)
    
    return means_df, ratio_df, approx_ratios_df