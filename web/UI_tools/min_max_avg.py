from matplotlib import pyplot as plt

from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class MinMaxAvg:

    def __init__(self, selected_tool: str, selected_label: str, selected_set: str, col3, col2, display: bool = True):
        self.__training_set = {}
        self.__selected_tool = selected_tool
        self.__selected_label = selected_label
        self.__selected_set = selected_set
        self.__col3 = col3
        self.__col2 = col2

        if display:
            max_label, min_label, avg_label = self.get_min_max_avg()
            self.create_results(selected_label, max_label, min_label, avg_label)

    def get_min_max_avg(self):
        if not TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [self.__selected_tool, self.__selected_label, self.__selected_set]
        ).is_cached():
            dr = DataReader()
            if self.__training_set == "Set A":
                self.__training_set = dr.training_setA
            elif self.__training_set == "Set B":
                self.__training_set = dr.training_setB
            else:
                self.__training_set = dr.combined_training_set

        analyse_tool = TrainingSet(
            "exploratory_data_analysis_min_max_avg",
            self.__training_set,
            [self.__selected_tool, self.__selected_label, self.__selected_set]
        )
        max_label = analyse_tool.get_max_for_label(self.__selected_label)

        min_label = analyse_tool.get_min_for_label(self.__selected_label)

        avg_label = analyse_tool.get_avg_for_label(self.__selected_label)
        return max_label, min_label, avg_label

    def create_results(self, selected_label, max_label, min_label, avg_label):
        fig, ax1 = plt.subplots()
        ax1.bar(['max', 'min', 'average'], height=[float(max_label[1]), float(min_label[1]), avg_label], color="g")
        ax1.set_title('Min, Max and average of ' + selected_label)
        self.__col2.pyplot(fig)
        self.__col3.metric("Max of " + selected_label, max_label[1])
        self.__col3.metric("Min of " + selected_label, min_label[1])
        self.__col3.metric("Average of " + selected_label, round(avg_label, 2))
