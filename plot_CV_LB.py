# use kaggle CLI to get CV and LB scores (CV in the submit message)

import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    COMPETITION = sys.argv[1]
except IndexError:
    COMPETITION = 'playground-series-s5e9'


result = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', COMPETITION],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                        )


lines = [line.split() for line in result.stdout.strip().split('\n')]

selected_columns = [1, 2, 3, 5, 7]
column_names = ['date', 'time', 'method', 'CV score', 'pLB score']

filtered_data = [[line[i] for i in selected_columns] for line in lines[2:]]
df = pd.DataFrame(filtered_data, columns=column_names)
df['CV score'] = df['CV score'].str.strip('()').astype('float')
df['pLB score'] = df['pLB score'].astype('float')


# Now you can plot
sc = sns.scatterplot(data=df, x='CV score', y='pLB score', hue='method')
plt.show()
