import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CSV_FILE = sys.argv[1]

reference = pd.DataFrame({
    'Label': ['28-28', '17-17-17', '10-26-26', 'DAP', '20-20', '14-35-14', 'Urea'],
    'public LB': [7264, 7384, 7808, 6299, 7359, 7606, 6180],
    'training data': [111158, 112453, 113887, 94860, 110889, 114436, 92317],
})

def make_cm_plot(df: pd.DataFrame) -> None:
    """ Plot confusion matrices 1st vs. 2nd and 1st vs. 3rd prediction label.
    Args:
        df (pd.DataFrame): From submission csv file.
    """
    cm_1_2 = confusion_matrix(df['1st'], df['2nd'], normalize='true')
    cm_1_3 = confusion_matrix(df['1st'], df['3rd'], normalize='true')

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_1_2, display_labels=class_names)
    disp.plot(ax=axs[0], cmap='Blues', values_format='.2f', xticks_rotation=45)
    axs[0].set_xlabel('2nd predicted label')
    axs[0].set_ylabel('1st predicted label')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_1_3, display_labels=class_names)
    disp.plot(ax=axs[1], cmap='Blues', values_format='.2f', xticks_rotation=45)
    axs[1].set_xlabel('3rd predicted label')
    axs[1].set_ylabel('1st predicted label')

    plt.tight_layout()
    plt.savefig(f"{CSV_FILE[:-4]}_mapk_confusion.png", bbox_inches='tight')
    plt.close()


def make_prediction_frequency_plot(df: pd.DataFrame) -> None:
    """ Plot frequencies of all predicted labels.
    Args:
        df (pd.DataFrame): From submission csv file.
    """
    melted_df = submitted_df.melt(value_vars=['1st', '2nd', '3rd'], var_name='Rank',
                                  value_name='Label')

    counts = melted_df.groupby(['Label', 'Rank']).size().reset_index(name='Count')
    counts['Proportion'] = counts.groupby('Rank')['Count'].transform(lambda x: x / x.sum())

    plt.figure(figsize=(5, 5))
    sns.barplot(data=counts, x='Label', y='Proportion', hue='Rank')
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('')
    plt.ylabel('Prediction frequency')

    plt.tight_layout()
    plt.savefig(f"{CSV_FILE[:-4]}_prediction_frequency.png", bbox_inches='tight')
    plt.close()


submitted_df = pd.read_csv(CSV_FILE)
submitted_df[['1st', '2nd', '3rd']] = submitted_df.iloc[:, 1].str.split(' ', expand=True)
class_names = sorted(pd.unique(pd.concat([submitted_df['1st'], submitted_df['2nd'],
                                          submitted_df['3rd']])))

make_cm_plot(submitted_df)
make_prediction_frequency_plot(submitted_df)
