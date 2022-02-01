from typing import Tuple

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
    # todo: fix index for new_x (patient_ids are turned into numbers by near_miss.fit_resample())
    # index = x_matrix.index.to_series(index=pd.Index(["patient_id"]))
    # index = x_matrix.index.to_series(name="patient_id")
    #
    # x_matrix = pd.concat([x_matrix, index], ignore_index=True)

    near_miss = NearMiss(version=version)
    new_x, new_y = near_miss.fit_resample(x_matrix, sepsis_df)

    # new_x.set_index(new_x["patient_id"], inplace=True)
    # new_x.drop(["patient_id"], inplace=True)

    return new_x, new_y


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

