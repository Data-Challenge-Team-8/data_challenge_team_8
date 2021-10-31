from IO.data_reader import DataReader

from tools.analyze_tools import AnalyzeTool

# this is the main file
if __name__ == '__main__':
    readData = DataReader()
    training_setA = readData.training_setA
    tool = AnalyzeTool(training_setA)

    results = tool.do_whole_training_set_analysis(export_to_csv=True)
