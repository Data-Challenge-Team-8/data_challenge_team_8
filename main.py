import os
from objects.training_set import TrainingSet
from tools.pacmap_analysis import PacmapAnalysis

if __name__ == '__main__':
    # Build Dashboard
    os.system("streamlit run "+os.path.join("web", "app.py"))

    # Test Pacmap Dimensionality Reduction
    # testing_set = TrainingSet.get_training_set(selected_label='none',
    # selected_tool='none',
    # selected_set='Set A')

    # print(testing_set.data)
    #
    # print("Elements:")
    # for i, element in enumerate(testing_set.data):
    #     print(i, element)
    #     if i == 10:
    #         break

    # testing_pacmap = PacmapAnalysis.get_analysis(testing_set)
    # testing_pacmap.plot_pacmap()
