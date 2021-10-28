from IO.data_reader import DataReader

import datetime


if __name__ == '__main__':
    readData = DataReader()

    start = datetime.datetime.now()
    print("Starting to read one patient data:", str(start))

    p = readData.get_patient_setA("p000095")

    end = datetime.datetime.now()
    print("Finished to read one patient data:", str(end))
    print("Diff:", str(end-start))

    print()
    print("Starting to read training set A:", str(start))

    training_setA = readData.training_setA

    end = datetime.datetime.now()
    print("Finished to read training set A:", str(end))
    print("Diff:", str(end - start))

