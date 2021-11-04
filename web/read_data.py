import pickle


class ReadData:
    PATH_MAX_MIN_AVG = 'data/min_max_avg.p'
    PATH_MISSING = 'data/missing_vals_all.p'

    def __init__(self):
        self.__min_max_avg_label = pickle.load(open(self.PATH_MAX_MIN_AVG, "rb"))
        self.__missing_val_label = pickle.load(open(self.PATH_MISSING, "rb"))

    @property
    def min_max_avg(self):
        return self.__min_max_avg_label

    @property
    def missing(self):
        return self.__missing_val_label

    def min_val_labels(self, label):
        min_val = self.__min_max_avg_label["min_vals"][label]
        return min_val

    def max_val_labels(self, label):
        max_val = self.__min_max_avg_label["max_vals"][label]
        return max_val

    def avg_val_labels(self, label):
        avg_val = self.__min_max_avg_label["avg_vals"][label]
        return avg_val

    def missing_val_labels(self, label):
        missing_val = self.__missing_val_label["missing_vals"][label]
        return missing_val
