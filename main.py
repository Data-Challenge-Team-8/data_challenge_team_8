from objects.training_set import TrainingSet
from classifier.timeseries.time_series_forest import TimeSeriesForest
from tools.visualization.time_series_comparison import plot_time_series_density

if __name__ == '__main__':
    set_a = TrainingSet.get_training_set("rnd Sample A")
    #set_a = TrainingSet.get_training_set("Set A")
    #avg_df = set_a.get_average_df(use_interpolation=True, fix_missing_values=True)
    #sepsis_df = set_a.get_sepsis_label_df()

    tsf = TimeSeriesForest(data_set=set_a, train_fraction=0.8, feature="HR")
    print("Starting setup ...")
    tsf.setup()
    print("Starting plotting ...")
    plot_time_series_density(tsf.train_data[0], label="HR", set_name=set_a.name+" (fixed, interpolated)")
    print("Start training & testing ...")
    tsf.train()

    tsf.display_confusion_matrix(plotting=True)
    print(":)")
