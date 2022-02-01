from typing import List
import matplotlib.pyplot as plt

from objects.training_set import TrainingSet


def plot_histogram_count(set: TrainingSet, features: List[str], label: str = "SepsisLabel", n_bins: int = None,
                         show_plot: bool = True, save_to_file: bool = False):
    """
    Plot histograms with all features as x-axis and the amount of label at the X as the height of the bar.
    :param set:
    :param features:
    :param label:
    :param n_bins: number of bins used, default None, if None the length of the current feature is used
    :param show_plot: flag if the plot will be showed by matplotlib
    :param save_to_file: flag if the plot will be saved to file
    :return:
    """
    fig, axs = plt.subplots(len(features))
    if not isinstance(axs, list):
        axs = [axs]

    label_data = set.get_feature(label)
    for i in range(len(features)):
        feature_data = set.get_feature(features[i])
        feature_series = feature_data.mean()

        n_bins = n_bins if n_bins is not None else len(range(int(feature_data.max().max()+1)))
        label_counts = []
        feature_counts = []
        for k in range(len(feature_series)):
            if label_data.loc[feature_series.index[k]] != 0:
                label_counts.append(int(feature_series[k]))
            feature_counts.append(int(feature_series[k]))

        print("feature counts", feature_counts)
        print("label counts", label_counts)
        print("n_bins", n_bins)
        axs[i].hist(feature_counts, bins=n_bins//2, label=features[i])
        axs[i].hist(label_counts, bins=n_bins//2, label=label)
        #axs[i].grid()
        axs[i].set_title(f"\"{label}\" distribution over \"{features[i]}\"")
        axs[i].legend()

    plt.show()
    plt.clf()
