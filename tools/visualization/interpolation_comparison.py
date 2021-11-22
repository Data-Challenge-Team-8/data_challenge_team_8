from IO.data_reader import FIGURE_OUTPUT_FOLDER
from objects.patient import Patient
from objects.training_set import TrainingSet

import os.path

import matplotlib.pyplot as plt
import pandas as pd


def plot_data_with_and_without_interpolation(plot_title: str, x_label: str, y_label: str, patient: Patient,
                                             label: str, interp_method: str, order: int = None,
                                             save_to_file: bool = False):
    """
    Example usage with a patients Heart Rate label
    :param label:
    :param patient:
    :param plot_title:
    :param x_label:
    :param y_label:
    :param interp_method:
    :param order:
    :param save_to_file:
    :return:
    """
    series_data = patient.data[label]
    interp_data = patient.get_interp_data(interp_method=interp_method, order=order)[label]

    if interp_data.isna().sum() == len(series_data):
        print("Warning: Tried to plot interpolation for discarded label! See Patient.NAN_DISMISSAL_THRESHOLD")
        return

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(plot_title + " (raw data)")
    line, = ax1.plot([i for i in range(len(series_data))], series_data, markevery=1, marker="H", mfc="grey", color='b',
                     label=f"{y_label}")

    ax2 = fig.add_subplot(3, 1, 3)
    ax2.set_ylabel(y_label)
    ax2.set_xlabel(x_label)
    if order is not None:
        ax2.set_title(plot_title + f"(interpolated, {interp_method}, order {order})")
    else:
        ax2.set_title(plot_title + f" (interpolated, {interp_method})")
    line2, = ax2.plot([i for i in range(len(interp_data))], interp_data, markevery=1, marker="H", mfc="r", color='r',
                      label=f"interpolated {y_label}", lw=2, ms=6)
    line3, = ax2.plot([i for i in range(len(series_data))], series_data, markevery=1, marker="H", mfc="grey", color='b',
                      label=f"{y_label}", lw=3, ms=4)
    plt.legend(bbox_to_anchor=(0.25, 1.35), loc="lower left")

    if not save_to_file:
        plt.show()
    else:
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.mkdir(FIGURE_OUTPUT_FOLDER)

        f = os.path.join(FIGURE_OUTPUT_FOLDER, plot_title.replace(" ", "_") + ".png")
        print(f"Saving figure \"{plot_title}\" to file {f}")
        plt.savefig(f)

    plt.close()


def plot_most_interesting_interpolation_patients(training_set: TrainingSet):

    def search_good_patients(training_set: TrainingSet, max_candidates: int = 10):
        candidates = []
        for patient in training_set.data.values():
            for label in Patient.DATA_LABELS:
                rel_nan = patient.data[label].isna().sum() / len(patient.data[label])
                diff = abs(rel_nan - 0.5)
                if diff < min_range:
                    candidates.append((patient, label, diff))

        candidates.sort(key=lambda x: x[2])
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        return candidates

    min_range = 0.1
    max_candidates = 10

    print("Searching most interesting patients & labels ...")
    candidates = search_good_patients(training_set, max_candidates)

    for best_patient, best_label, _ in candidates:
        print(f"Starting to plot {best_label} for Patient {best_patient.ID}")
        plot_data_with_and_without_interpolation(f"{best_label} Interpolation ({best_patient.ID})", "time",
                                                 y_label=best_label, label=best_label,
                                                 patient=best_patient, interp_method="quadratic", save_to_file=True)
