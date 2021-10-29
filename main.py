from IO.data_reader import DataReader

import datetime
from objects.analyse_tools import AnalyseTool

# this is the main file
if __name__ == '__main__':
    readData = DataReader()
    training_setA = readData.training_setA
    tool = AnalyseTool(training_setA)

    start = datetime.datetime.now()
    print("Starting to read one patient data:", str(start))

    p = readData.get_patient_setA("p000095")

    end = datetime.datetime.now()
    print("Finished to read one patient data:", str(end))
    print("Diff:", str(end - start))

    print()
    print("Starting to read training set A:", str(start))

    print("Missing values for p000095: ", tool.missing_values_single("pH", "p000095"))
    print("Single user min value for p000095: ", tool.min_single("HR", "p000095"))
    print("Single user min value for p000095: ", tool.max_single("HR", "p000095"))
    print("Single user min value for p000095: ", tool.avg_single("HR", "p000095"))
    print()

    end = datetime.datetime.now()
    print("Finished to read training set A:", str(end))
    print("Diff:", str(end - start))
