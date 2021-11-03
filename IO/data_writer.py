from tools.analyze_tools import AnalyzeTool
import pandas as pd

class DataWriter:

    def __init__(self, training_set) -> None:
        self.tool = AnalyzeTool()
        self.__training_set = training_set

    def save_all_min_max_avg_missing_value(self):
        data = dict()



