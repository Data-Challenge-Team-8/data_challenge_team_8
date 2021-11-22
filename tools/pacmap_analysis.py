from objects.training_set import TrainingSet

import pacmap
from matplotlib import pyplot as plt


def calculate_pacmap(training_set: TrainingSet):
    """
    Calculate a PaCMAP transformation of the given TrainingSet.

    Based on TrainingSet.get_average_df() and the fix_missing_values flag
    :param training_set:
    :return: data as returned by the PaCMAP algorithm
    """
    avg_df = training_set.get_average_df(fix_missing_values=True)
    avg_np = avg_df.transpose().to_numpy()

    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    embedding = pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, verbose=True)
    data_transformed = embedding.fit_transform(avg_np, init="pca")

    return data_transformed


def plot_pacmap(data):
    """
    Plots the given PaCMAP data using matplotlib
    :param data:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.scatter(data[:, 0], data[:, 1], cmap="Spectral", c="r", s=0.6)

    plt.show()
    plt.close()
