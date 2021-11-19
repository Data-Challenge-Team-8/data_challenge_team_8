import os
from datetime import datetime

from objects.training_set import TrainingSet
from tools.pacmap_analysis import PacmapAnalysis

if __name__ == '__main__':
    # Build Dashboard
    # os.system("streamlit run "+os.path.join("web", "app.py"))

    # test caching time
    # start = datetime.now()
    # print("Starting with cache: ", start)
    # first_set_A = TrainingSet.get_training_set('Set A')
    # end = datetime.now()
    # print("Finished: ", end)
    # print("Difference: ", end - start)

    # test label stats
    test_set_A = TrainingSet.get_training_set('Set A')
    labels_average, labels_std_dev, labels_rel_NaN = test_set_A.calc_stats_for_labels()

    print("Dict of Label Averages: ", labels_average)
    print("Dict of Label Standard Deviation: ", labels_std_dev)
    print("Dict of Label NaN: ", labels_rel_NaN)


    # TODO: Use Averages, std_dev for Pacmap (maybe use label_nan to sort out unnecessary labels or do interpolation)
    # df_avg = test_set_A.get_dataframe_averages()
    # print(df_avg)
    # test_pacmap = PacmapAnalysis.get_analysis(test_set_A)
    # test_pacmap.plot_pacmap()
