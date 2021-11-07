from IO.data_reader import DataReader
from objects.training_set import TrainingSet
from web.UI_tools.plot_label_to_sepsis import PlotLabelToSepsis
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis


class DataLoader:

    def __init__(self):
        pass

    @staticmethod
    def load_before(sets, options):
        dr = DataReader()
        set_a = dr.training_setA
        set_b = dr.training_setB
        set_a_b = dr.training_setA
        for s in sets:
            if s == 'Set A':
                for o in options:
                    if o == 'Min, Max, Average':
                        for label in ExploratoryDataAnalysis(display=False).LABELS:
                            analyse_tool = TrainingSet(
                                "exploratory_data_analysis_min_max_avg",
                                set_a,
                                ['Min, Max, Average', label, s]
                            )
                            analyse_tool.get_max_for_label(label)
                            analyse_tool.get_min_for_label(label)
                            analyse_tool.get_avg_for_label(label)
                    elif o == 'Plots':
                        for label in PlotLabelToSepsis('Mathematical statistics', display=False).LABELS:
                            analyse_tool = TrainingSet(
                                "mathematical_statistics",
                                set_a,
                                ['Mathematical statistics', label, s]
                            )
                            analyse_tool.get_plot_label_to_sepsis(label)
                    elif 0 == 'Missing Values':
                        for label in ExploratoryDataAnalysis(display=False).LABELS:
                            analyse_tool = TrainingSet(
                                "exploratory_data_analysis_missing_values",
                                set_a,
                                ['Min, Max, Average', label, s]
                            )
                            analyse_tool.get_avg_rel_NaN_amount_for_label(label)

            elif s == 'Set B':
                if o == 'Min, Max, Average':
                    for label in ExploratoryDataAnalysis(display=False).LABELS:
                        analyse_tool = TrainingSet(
                            "exploratory_data_analysis_min_max_avg",
                            set_a,
                            ['Min, Max, Average', label, s]
                        )
                        analyse_tool.get_max_for_label(label)
                        analyse_tool.get_min_for_label(label)
                        analyse_tool.get_avg_for_label(label)
                elif o == 'Plots':
                    for label in PlotLabelToSepsis('Mathematical statistics', display=False).LABELS:
                        analyse_tool = TrainingSet(
                            "mathematical_statistics",
                            set_a,
                            ['Mathematical statistics', label, s]
                        )
                        analyse_tool.get_plot_label_to_sepsis(label)
                elif 0 == 'Missing Values':
                    for label in ExploratoryDataAnalysis(display=False).LABELS:
                        analyse_tool = TrainingSet(
                            "exploratory_data_analysis_missing_values",
                            set_a,
                            ['Min, Max, Average', label, s]
                        )
                        analyse_tool.get_avg_rel_NaN_amount_for_label(label)
            elif s == 'Set A + B':
                if o == 'Min, Max, Average':
                    for label in ExploratoryDataAnalysis(display=False).LABELS:
                        analyse_tool = TrainingSet(
                            "exploratory_data_analysis_min_max_avg",
                            set_a,
                            ['Min, Max, Average', label, s]
                        )
                        analyse_tool.get_max_for_label(label)
                        analyse_tool.get_min_for_label(label)
                        analyse_tool.get_avg_for_label(label)
                elif o == 'Plots':
                    for label in PlotLabelToSepsis('Mathematical statistics', display=False).LABELS:
                        analyse_tool = TrainingSet(
                            "mathematical_statistics",
                            set_a,
                            ['Mathematical statistics', label, s]
                        )
                        analyse_tool.get_plot_label_to_sepsis(label)
                elif 0 == 'Missing Values':
                    for label in ExploratoryDataAnalysis(display=False).LABELS:
                        analyse_tool = TrainingSet(
                            "exploratory_data_analysis_missing_values",
                            set_a,
                            ['Min, Max, Average', label, s]
                        )
                        analyse_tool.get_avg_rel_NaN_amount_for_label(label)
