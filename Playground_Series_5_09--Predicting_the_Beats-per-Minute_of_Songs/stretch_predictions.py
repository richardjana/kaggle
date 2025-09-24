import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

# test hypothesis: most models play it safe — their predictions fall back almost entirely to the
# mean. While train.csv has BPM values ranging from 46 to 206, most submission files sit cramped
# between 115 and 130.
# Try to find the optimal stretch factor?

def stretch_data(data: pd.Series, factor: float) -> pd.Series:
    y_mean = data.mean()
    return y_mean + (data - y_mean) * np.exp(factor)

def make_diagonal_plot(original, stretched, metric, metric_name, fname, precision = 5):
    chart = sns.scatterplot(data=original, x='oof', y='y_true', alpha=0.25)
    sns.scatterplot(data=stretched, x='oof', y='y_true', alpha=0.25)

    min_val = min(chart.get_xlim()[0], chart.get_ylim()[0])
    max_val = max(chart.get_xlim()[1], chart.get_ylim()[1])
    chart.set_xlim([min_val, max_val])
    chart.set_ylim([min_val, max_val])
    chart.plot([min_val, max_val], [min_val, max_val], linewidth=1, color='k')

    chart.set_aspect('equal')

    metric_value = metric(original['oof'], original['y_true'])
    labels = [f"original ({metric_value:.{precision}f})"]
    metric_value = metric(stretched['oof'], stretched['y_true'])
    labels += [f"stretched ({metric_value:.{precision}f})"]
    plt.legend(labels=labels, title=f"dataset ({metric_name}):", loc='best')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()


oof = pd.read_csv('oof.csv')



LOW = 0.00
HIGH = 0.15
factors = np.linspace(LOW, HIGH, int((HIGH-LOW)/0.01+1))
rmses = []
for fac in factors:
    oof_stretched = oof.copy()
    oof_stretched['oof'] = stretch_data(oof['oof'], fac)
    rmses.append(root_mean_squared_error(oof['y_true'], oof_stretched['oof']))
    #print(fac, root_mean_squared_error(oof['y_true'], oof_stretched['oof']))

plt.figure(figsize=(8, 4))
plt.plot(factors, rmses)
plt.show()

# find best
i_best = np.argmin(rmses)
oof_stretched = oof.copy()
oof_stretched['oof'] = stretch_data(oof['oof'], factors[i_best])

make_diagonal_plot(oof, oof_stretched, root_mean_squared_error, 'RMSE', 'stretch_plot.png')

#submission = pd.read_csv('predictions_optuna.csv')
#submission['BeatsPerMinute'] = y_mean + (submission['BeatsPerMinute'] - y_mean) * np.exp(stretch_factor)
#submission.to_csv('stretched.csv', index=False)
