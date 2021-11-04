
import pickle


class DataWriter:
    """ This class writes data to a pickle file """

    def __init__(self) -> None:
        return

    @staticmethod
    def write_min_max_avg(min_vals, max_vals, avg_vals) -> None:
        """ write the min, max and avg values for each label to a pickle """
        data = {"min_vals": min_vals, "max_vals": max_vals, "avg_vals": avg_vals}
        pickle.dump(data, open("data/min_max_avg.p", "wb"))

    @staticmethod
    def write_missing_val(missing_vals) -> None:
        """ write the amount of missing values for each label to a pickle """
        data = {"missing_vals": missing_vals}
        pickle.dump(data, open("data/missing_vals_all.p", "wb"))