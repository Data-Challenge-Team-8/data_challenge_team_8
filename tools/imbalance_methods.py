from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from pandas import DataFrame

from objects.training_set import TrainingSet

def get_near_miss_for_training_set(training_set: TrainingSet, version: int = 2) -> Tuple[DataFrame, DataFrame]:
    """This can be used to do balanced method NearMiss on complete TS."""
    avg_df = training_set.get_average_df(use_interpolation=True, fix_missing_values=True)
    sepsis_df = training_set.get_sepsis_label_df()
    x_matrix = avg_df.transpose()

    index_numbers_list = list(range(0, len(x_matrix)))
    temp_index_strings = x_matrix.index.to_series(name="patient_id_strings", index=index_numbers_list)
    x_matrix["patient_id_number"] = np.arange(len(x_matrix))
    near_miss = NearMiss(version=version)
    new_x, new_y = near_miss.fit_resample(x_matrix, sepsis_df)         # problem: all values in x_matrix must be int or float for nearmiss (stupid but thats how it is)

    # get mapping of kept id_number to refill index of new_X with patient_id from temp_index_strings
    list_of_kept_index = new_x["patient_id_number"]
    patient_ids = temp_index_strings.iloc[list_of_kept_index.tolist()]        # kann man hier auch direkt die letzte spalte entfernen?
    new_x.set_index(patient_ids, inplace=True)
    new_x.drop(["patient_id_number"], axis=1, inplace=True)

    return new_x, new_y     # careful: new_x has patient_ids as index, new_y has numbers as index


def get_near_miss_for_split_data(x_matrix, y_label, version: int = 2) -> Tuple[DataFrame, DataFrame]:
    """This can be used to do balanced method NearMiss on avg_df and sepsis_label."""
    near_miss = NearMiss(version=version)
    new_x, new_y = near_miss.fit_resample(x_matrix, y_label)

    return new_x, new_y


def get_smote_for_split_data(x_matrix, y_label, random_state: int = 1337) -> Tuple[DataFrame, DataFrame]:
    """This can be used to do balanced method SMOTE on avg_df and sepsis_label."""
    smote = SMOTE(random_state=random_state)
    new_x, new_y = smote.fit_resample(x_matrix, y_label)

    return new_x, new_y

