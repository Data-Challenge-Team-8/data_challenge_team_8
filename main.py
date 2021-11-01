from IO.data_reader import DataReader

from tools.analyze_tools import AnalyzeTool

# this is the main file
if __name__ == '__main__':
    readData = DataReader()
    training_setA = readData.training_setA
    tool = AnalyzeTool(training_setA)
    #tool.do_basic_set_analysis(print_to_stdout=True)
    tool.do_whole_training_set_analysis(export_to_csv=True)
    tool = AnalyzeTool(readData.training_setB)
    tool.do_whole_training_set_analysis(export_to_csv=True)

    exit(0)
    print("The subset of all Patients with HR between 90 and 100")
    print(len(tool.subset_all('HR', 90, 100)))
    print("The subset of all gender with val 1")
    print(len(tool.subset_all('gender', 1, 1)))
    print("The subset of all patient with age between 18 and 21")
    print(len(tool.subset_all('age', 18, 21)))
    results = tool.do_whole_training_set_analysis(export_to_csv=False)
