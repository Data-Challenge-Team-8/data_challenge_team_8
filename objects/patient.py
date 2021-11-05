import pandas as pd
import numpy as np


class NotUniqueIDError(Exception):
    pass


class Patient:
    """
    Class containing and allowing access to a whole patients data.

    Important: each row will represent a single hour's worth of data
    """

    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]
    FEMALE = 0
    MALE = 1

    patient_id_set = set()  # for checking uniqueness in ID

    def __init__(self, patient_ID: str, patient_data: pd.DataFrame):
        self.__data = patient_data
        self.__patient_ID: str = None

        if patient_ID not in Patient.patient_id_set:
            self.__patient_ID = patient_ID
            Patient.patient_id_set.add(patient_ID)
        else:
            raise NotUniqueIDError(f"Patient ID \"{patient_ID}\" was not unique!")

        for label in Patient.LABELS:  # Sanity check of expected labels against present labels in data
            if label not in self.data.columns.values:
                raise ValueError("Given patient data did not match the expected format! Please check the columns.")

    def __del__(self):  # removing ID from the set, so it can be given out again
        if self.__patient_ID in Patient.patient_id_set:
            Patient.patient_id_set.remove(self.__patient_ID)

    @property
    def ID(self) -> str:
        """
        Return a unique ID for the patient
        :return:
        """
        return self.__patient_ID

    @property
    def data(self):
        return self.__data

    def get_standard_deviation(self, label: str) -> float:
        """
        Get the standard deviation for a given label (see Patient.LABELS)
        :param label:
        :return:
        """
        a = self.data[label].dropna().to_numpy()
        return np.std(a)

    #################### Vital Signs ######################

    @property
    def vital_signs(self) -> pd.DataFrame:
        """
        Data Frame of all available vital signs.
        :return:
        """
        return self.data[["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]]

    @property
    def HR(self) -> pd.Series:
        """
        Heart Rate (beats per minute)
        :return:
        """
        return self.data["HR"]

    @property
    def O2Sat(self) -> pd.Series:
        """
        Pulse oximetry (in percentage %)
        :return:
        """
        return self.data["O2Sat"]

    @property
    def Temp(self) -> pd.Series:
        """
        Body Temperature in °C
        :return:
        """
        return self.data["Temp"]

    @property
    def SBP(self) -> pd.Series:
        """
        Systolic Blood Pressure (mmHg | millimeter of Mercury @ 0°C)

        Systolic blood pressure indicates how much pressure the blood is exerting against the artery walls when the
        heart beats pumping blood out. Systolic blood pressure is the top number of the blood pressure reading.

        Normal is less than 120mmHg (120/80)

        1 pascal is equal to 0.0075006156130264 mmHg
        :return:
        """
        return self.data["SBP"]

    @property
    def DBP(self) -> pd.Series:
        """
        Diastolic Blood Pressure (mmHg | millimeter of Mercury @ 0°C)

        Diastolic blood pressure indicates how much pressure the blood is exerting against the artery walls when the
        heart is relaxed between beats. Diastolic blood pressure is the bottom number of the blood pressure reading.

        Normal is less than 80mmHg (120/80)

        1 pascal is equal to 0.0075006156130264 mmHg
        :return:
        """
        return self.data["DBP"]

    @property
    def MAP(self) -> pd.Series:
        """
        Mean arterial pressure (mmHg | millimeter of Mercury @ 0°C)

        The mean arterial pressure is the average blood pressure in an individual
        during a single cardiac cycle.

        Normal is between 65 and 110 mmHg

        1 pascal is equal to 0.0075006156130264 mmHg
        :return:
        """
        return self.data["MAP"]

    @property
    def Resp(self) -> pd.Series:
        """
        Respiration rate (breaths per minute)

        Normal depends on age:
        Newborn: 30-60 breaths per minute
        Infant (1 to 12 months): 30-60 breaths per minute
        Toddler (1-2 years): 24-40 breaths per minute
        Preschooler (3-5 years): 22-34 breaths per minute
        School-age child (6-12 years): 18-30 breaths per minute
        Adolescent (13-17 years): 12-16 breaths per minute

        :return:
        """
        return self.data["Resp"]

    @property
    def EtCO2(self) -> pd.Series:
        """
        End tidal carbon dioxide (mmHg | millimeter of Mercury @ 0°C)

        End-Tidal Carbon Dioxide refers to the partial pressure or concentration of carbon dioxide (CO2) at the end
        of exhalation.

        Normal is between 35-45mmHg
        :return:
        """
        return self.data["EtCO2"]

    ################# Laboratory Values ###################
    @property
    def LabValues(self) -> pd.DataFrame:
        """
        Return a dataframe of all laboratory values
        :return:
        """
        return self.data[["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
                          "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
                          "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
                          "Platelets"]]

    @property
    def BaseExcess(self) -> pd.Series:
        """
        Measure of excess bicarbonate (mmol/L)

        The base excess is the amount of strong acid that is required to return a patient's sample to a pH of 7.4,
        pCO₂ (partial pressure of CO2) to 40mmHg and temperature to 37°C. The base excess is an indicator of a
        metabolic process that is independent of the acid-base buffering system.

        Normal is between -2 to +2

        derivable from pH and pCO2 (https://acutecaretesting.org/en/journal-scans/understanding-base-excess-a-review-article)
        :return:
        """
        return self.data["BaseExcess"]

    @property
    def HCO3(self) -> pd.Series:
        """
        Bicarbonate (mmol/L)

        The bicarbonate level is significantly influenced by the acid-base buffering system, and can be affected by the
        presence of a respiratory process. Bicarbonate (HCO3^-) is a vital component of the pH buffering system of the
        human body (maintaining acid–base homeostasis).

        derivable from pH and pCO2 (https://acutecaretesting.org/en/journal-scans/understanding-base-excess-a-review-article)
        :return:
        """
        return self.data["HCO3"]

    @property
    def FiO2(self) -> pd.Series:
        """
        Fraction of inspired Oxygen (%)

        FiO2 is the molar or volumetric fraction of oxygen in the inhaled gas. FiO2 is used to represent the percentage
        of oxygen participating in gas-exchange. If the barometric pressure changes, the FiO2 may remain constant
        while the partial pressure of oxygen changes with the change in barometric pressure.

        Normal (for earth atmosphere) is 21%, medicine might use up to 100% but typically 50%

        :return:
        """
        return self.data["FiO2"]

    @property
    def pH(self) -> pd.Series:
        """
        pH (in pH)

        Normal is 7.4 (for Humans)
        Range 0 - 14
        :return:
        """
        return self.data["pH"]

    @property
    def PaCO2(self) -> pd.Series:
        """
        Partial pressure of carbon dioxide from arterial blood (mmHg | millimeter of Mercury @ 0°C)

        It often serves as a marker of sufficient alveolar ventilation within the lungs.

        Normal is 35 - 45 mmHg
        :return:
        """
        return self.data["PaCO2"]

    @property
    def SaO2(self) -> pd.Series:
        """
        Oxygen saturation from arterial blood (%)

        Normal is 100-95% (for adults)
        :return:
        """
        return self.data["SaO2"]

    @property
    def AST(self) -> pd.Series:
        """
        Aspartate transaminase (IU/L | International Units per Liter)

        AST is an enzyme that catalyzes the reversible transfer of an α-amino group between aspartate and glutamate
        and, as such, is an important enzyme in amino acid metabolism.

        Important: International Units (IU/L) is a substance dependent unit and is different to Units/L (U/L).
        Reference range can vary by laboratory.
        :return:
        """
        return self.data["AST"]

    @property
    def BUN(self) -> pd.Series:
        """
        Blood urea nitrogen (mg/dL)

        The liver produces urea in the urea cycle as a waste product of the digestion of protein.

        Normal is 6-20 mg/dL (2.1 to 7.1 mmol/L).

        Reference range can vary by laboratory.
        :return:
        """
        return self.data["BUN"]

    @property
    def Alkalinephos(self) -> pd.Series:
        """
        Alkaline phosphatase (IU/L)

        An Enzyme. In humans, it is found in many forms depending on its origin within the body. It plays an integral
        role in metabolism within the liver and development within the skeleton.

        Important: International Units (IU/L) is a substance dependent unit and is different to Units/L (U/L).
        Reference range can vary by laboratory.
        :return:
        """
        return self.data["Alkalinephos"]

    @property
    def Calcium(self) -> pd.Series:
        """
        Calcium (mg/dL)

        Calcium is an essential element needed in large quantities. The Ca2+ ion acts as an electrolyte and is vital to
        the health of the muscular, circulatory, and digestive systems; is indispensable to the building of bone;
        and supports synthesis and function of blood cells.

        Normal is 8.6-10.3 mg/dL (within serum/blood)
        :return:
        """
        return self.data["Calcium"]

    @property
    def Chloride(self) -> pd.Series:
        """
        Chloride (mmol/L)

        Chloride plays a role in regulation of osmotic pressure, electrolyte balance and acid-base homeostasis.
        It is present in all body fluids, and is the most abundant extracellular anion which accounts for around one
        third of extracellular fluid's tonicity.

        Normal is 97-106 mmol/L (for Adults)

        Reference range can vary by laboratory.
        :return:
        """
        return self.data["Chloride"]

    @property
    def Creatinine(self) -> pd.Series:
        """
        Creatinine (mg/dL)

        Measuring serum creatinine is the most commonly used indicator of renal function. An increase in
        serum creatinine can also be due to increased ingestion of cooked meat or increased muscle breakdown due to
        intense exercise as well as some medications.

        Normal is 0.6-1.3 mg/dL
        :return:
        """
        return self.data["Creatinine"]

    @property
    def Bilirubin_direct(self) -> pd.Series:
        """
        Bilirubin direct (mg/dL)

        Bilirubin is a substance made when your body breaks down old red blood cells. It is also part of bile,
        which your liver makes to help digest food. In the liver, bilirubin is changed into a form that your body can
        get rid of. This is called conjugated bilirubin or direct bilirubin. This bilirubin travels from the liver into
        the small intestine and in small amounts into the kidneys for excretion.
        (Gives urine its distinctive yellow color)

        Normal is 0-0.2 mg/dL (for adults)
        :return:
        """
        return self.data["Bilirubin_direct"]

    @property
    def Bilirubin_total(self) -> pd.Series:
        """
        Total bilirubin (mg/dL)

        The liver makes bile to help you digest food, and bile contains bilirubin. Most bilirubin comes from the
        body's normal process of breaking down old red blood cells. A healthy liver can normally get rid of bilirubin.

        Normal is <1 mg/dL
        :return:
        """
        return self.data["Bilirubin_total"]

    @property
    def Glucose(self) -> pd.Series:
        """
        Serum Glucose (mg/dL)

        Glucose makes for most of the cells in the body the major source of energy. Severe stress, recent stroke or
        trauma are likely to temporarily increase serum glucose.

        Normal is <100 mg/dL; >126 mg/dL indicates diabetes
        :return:
        """
        return self.data["Glucose"]

    @property
    def Lactate(self) -> pd.Series:
        """
        Lactic acid (mg/dL)

        In animals, L-lactate is constantly produced from pyruvate via the enzyme lactate dehydrogenase (LDH) in
        a process of fermentation during normal metabolism and exercise. It does not increase in concentration until
        the rate of lactate production exceeds the rate of lactate removal, typically during intense exercise.
        Levels vary greatly between arterial and venous blood.

        Normal is 4.5-14.4 mg/dL (arterial, more commonly tested)
                  4.5-19.8 mg/dL (venous)
        :return:
        """
        return self.data["Lactate"]

    @property
    def Magnesium(self) -> pd.Series:
        """
        Magnesium (mmol/L)

        The important interaction between phosphate and magnesium ions makes magnesium essential to the basic
        nucleic acid chemistry of all cells of all known living organisms. More than 300 enzymes require magnesium ions
        for their catalytic action, including all enzymes using or synthesizing ATP and those that use other
        nucleotides to synthesize DNA and RNA.

        Normal is 0.65-1.05 mmol/L (Adults)
                  0.72-0.9  mmol/L (Children)
        :return:
        """
        return self.data["Magnesium"]

    @property
    def Phosphate(self) -> pd.Series:
        """
        Phosphate (mg/dL)

        Phosphorus works together with the mineral calcium to build strong bones and teeth. Normally, the kidneys
        filter and remove excess phosphate from the blood. If phosphate levels in your blood are too high or too low,
        it can be a sign of kidney disease or other serious disorder.

        Normal is 3-4.5 mg/dL (Adults)
                4.5-6.5 mg/dL (Children)
                4.3-9.3 mg/dL (Newborn)
        :return:
        """
        return self.data["Phosphate"]

    @property
    def Potassium(self) -> pd.Series:
        """
        Potassium 	(mmol/L)

        Potassium plays a major role in the functioning of your muscles, nerves, digestive system and bodily organs
        such as your heart and kidneys. High levels can lead to, for example, cardiac arrest.

        Normal is 3.5-5.2 mmol/L
        :return:
        """
        return self.data["Potassium"]

    @property
    def TroponinI(self) -> pd.Series:
        """
        Troponin I (ng/mL)

        Troponin I is a cardiac and skeletal muscle protein family. It plays a part in muscle contraction
        The letter I is given due to its inhibitory
        character. Levels that exceed the normal range are strongly suggestive of cardiac injury.

        Normal is 0.3 ng/mL; >0.3 ng/mL indicates cardiac injury,
        >0.4 ng/mL indicates usually myocardial infarction (for example sepsis)
        :return:
        """
        return self.data["TroponinI"]

    @property
    def Hct(self) -> pd.Series:
        """
        Hematocrit (%)

        Hematocrit is the volume percentage of red blood cells in blood. The result can be affected by
        living at higher altitudes, pregnancy, significant recent blood loss, recent blood transfusion or
        severe dehydration.

        Normal is 40.7%-50.3% (for males)
                  36.1%-44.3% (for females)

        :return:
        """
        return self.data["Hct"]

    @property
    def Hgb(self) -> pd.Series:
        """
        Hemoglobin (g/dL)

        Hemoglobin is the iron-containing oxygen-transport metalloprotein in the red blood cells of
        almost all vertebrates.

        Normal is 13.5 - 17.5 g/dL (for males)
                  12.0 - 15.5 g/dL (for females)
        :return:
        """
        return self.data["Hgb"]

    @property
    def PTT(self) -> pd.Series:
        """
        partial thromboplastin time (seconds)

        Partial thromboplastin time measures the overall speed at which blood clots.

        Normal is 30s - 50s
        :return:
        """
        return self.data["PTT"]

    @property
    def WBC(self) -> pd.Series:
        """
        Leukocyte count (count*10^3/µL) (White Blood Cell)

        Leukocytes are the cells of the immune system that are involved in protecting the body against
        both infectious disease and foreign invaders. An increase above normal levels can indicate an immune response.

        Normal is 4*10^9/L - 1.1*10^10/L
                  4 * 10^3/µL - 11 * 10^3/µL
        :return:
        """
        return self.data["WBC"]

    @property
    def Platelets(self) -> pd.Series:
        """
        Platelets (count*10^3/µL)

        Platelets, also called thrombocytes, are a component of blood whose function is to react to bleeding
        from blood vessel injury by clumping, thereby initiating a blood clot.

        Normal is 150 * 10^3/µL - 400 * 10^3/µL
        :return:
        """
        return self.data["Platelets"]

    @property
    def Fibrinogen(self) -> pd.Series:
        """
        Fibrinogen (mg/dL)

        During tissue and vascular injury, it is converted enzymatically by thrombin to fibrin and then to a
        fibrin-based blood clot. Fibrin clots function primarily to occlude blood vessels to stop bleeding.

        Normal is 200-400 mg/dL (for adults)
        :return:
        """
        return self.data["Fibrinogen"]

    ################## Demographics ##################
    @property
    def demographics(self) -> pd.DataFrame:
        """
        Return a dataframe of all demographics
        :return:
        """
        return self.data[["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]]

    @property
    def age(self) -> pd.Series:
        """
        Age of the patient (years)

        Important: 100 for patients 90 or above

        :return:
        """
        return self.data["Age"]

    @property
    def gender(self) -> pd.Series:
        """
        Gender of the patient.

        See Patient.MALE and Patient.FEMALE for interpretation

        :return:
        """
        return self.data["Gender"]

    @property
    def unit1(self) -> pd.Series:
        """
        Administrative Identifier for ICU unit (MICU)

        :return:
        """
        return self.data["Unit1"]

    @property
    def unit2(self) -> pd.Series:
        """
        Administrative Identifier for ICU unit (SICU)

        :return:
        """
        return self.data["Unit2"]

    @property
    def hosp_adm_time(self) -> pd.Series:
        """
        Hours between hospital admit and ICU admit

        :return:
        """
        return self.data["HospAdmTime"]

    @property
    def ICULOS(self) -> pd.Series:
        """
        ICU length-of-stay (hours since ICU admit)
        :return:
        """
        return self.data["ICULOS"]

    @property
    def sepsis_label(self) -> pd.Series:
        """
        Label for Sepsis

        For sepsis patients, SepsisLabel is 1 if t >= t_sepsis -6 and t < t_sepsis - 6
        For non-sepsis patients, SepsisLabel is 0

        See "Challenge Data" at https://physionet.org/content/challenge-2019/1.0.0/ for further details
        :return:
        """
        return self.data["SepsisLabel"]
