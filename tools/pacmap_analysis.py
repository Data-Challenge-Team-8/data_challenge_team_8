import pacmap
from matplotlib import pyplot as plt


# TODO: Import training set, transform to dataframe, dimensionality reduction -> return new dataframe, plot the new df
class PacmapAnalysis:

    def __init__(self, training_set):
        self.training_set = training_set(selected_label='none',
                                         selected_tool='none',
                                         selected_set='Set A')
        self.calculate_pacmap()

    @classmethod
    def get_pacmap_analysis(cls, training_set):
        # if cached: return cached analysis-values ?
        # else:
        new_analysis = PacmapAnalysis(training_set)
        return new_analysis

    def calculate_pacmap(self):
        old_dataframe = self.training_set.transform_into_df()
        embedding = pacmap.PaCMAP(n_dims=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
        X_transformed = embedding.fit_transform(old_dataframe.values, init="pca")

        self.plot_pacmap(X_transformed)

    def plot_pacmap(self, new_dataframe):
        pass
