import pandas as pd


class Label:
    """
    Class for all 40 Labels
    """

    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]
    FEMALE = 0
    MALE = 1

    def __init__(self, name: str):
        self.name = name
        # self.__min_for_label: Dict[str, Tuple[str, float]] = {}
        # self.__max_for_label: Dict[str, Tuple[str, float]] = {}
        # self.__avg_for_label: Dict[str, float] = {}
        # self.__NaN_amount_for_label: Dict[str, int] = {}
        # self.__non_NaN_amount_for_label: Dict[str, int] = {}
        # self.__plot_label_to_sepsis: Dict[str, Tuple[List[float], List[float]]] = {}
        # self.__min_data_duration: Tuple[str, int] = None
        # self.__max_data_duration: Tuple[str, int] = None
        # self.__avg_data_duration: float = None
        # self.__sepsis_patients: List[str] = None