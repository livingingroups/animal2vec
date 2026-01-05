import argparse
import ast
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def get_lbl_path_from_wav_path(wav_path: Path) -> Path:
    lbl_path = str(wav_path.with_suffix(".h5"))
    lbl_path = Path(lbl_path.replace("/wav/", "/lbl/"))
    return lbl_path


def remove_empty_rows(
    predictions: NDArray, targets: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    When saving the predictions, the predictions and targets are interpolated with a lot of empty predictions.
    This makes the predictions and the targets very large, and slows down the calculation of the average precision. 
    This function removes rows where all target values are zero; this doesn't affect the average precision calculation.
    This assumes that the targets have at least one positive class for each audio clip (eg row).
    
    Args:
        predictions (NDArray): Array of shape (num_segments, num_classes) containing the predictions.
        targets (NDArray): Array of shape (num_segments, num_classes) containing target labels.
    Returns:
        Tuple[NDArray, NDArray]: Filtered predictions and targets with empty rows removed.

    """
    empty_indices_targets = np.where(targets.sum(axis=1) == 0)[0]
    empty_indices_pred = np.where(predictions.sum(axis=1) == 0)[0]
    empty_indices = np.intersect1d(empty_indices_pred, empty_indices_targets)
    all_indices = np.arange(len(targets))
    non_empty_indices = np.setdiff1d(all_indices, empty_indices)
    return predictions[non_empty_indices], targets[non_empty_indices]


def get_predictions(
    predictions_path: str, segmented: bool = True
) -> Tuple[NDArray, NDArray]:
    """
    Reads the predictions and targets from the h5 created by the get_results_for_single_manifest.py script.
    Args:
        predictions_path (str): Path to the h5 file containing predictions and targets.
        segmented (bool): Whether to read segmented predictions/targets or framewise ones.
    Returns:
        Tuple[NDArray, NDArray]: Arrays containing predictions and targets.
    """
    predictions = []
    targets = []

    with h5py.File(predictions_path, "r") as h5_file:
        for event in tqdm(h5_file):
            if segmented:
                predictions.append(h5_file[event]["segmented_likelihood"])
                targets.append(h5_file[event]["segmented_target"])
            else:
                predictions.append(h5_file[event]["likelihood"])
                targets.append(h5_file[event]["target"])
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = np.array(targets)
        targets = targets.reshape(
            -1, targets.shape[-1]
        )  # shape (num_segments, num_classes)
    predictions, targets = remove_empty_rows(predictions, targets)
    return predictions, targets


def get_tot_duration_vocalizations(targets: NDArray, targets_sample_rate: int) -> List[float]:
    """
    Calculate the total duration of vocalizations in minutes.

    Args:
        targets (NDArray): Array containing target labels.
        target_sample_rate (int): Sample rate of the targets array.

    Returns:
        List[float]: The list of the total durations of all the vocalization types in minutes.
    """
    num_vocalization_frames = np.sum(targets, axis=0)
    total_duration_min = num_vocalization_frames / (targets_sample_rate * 60)
    return np.round(total_duration_min, 1).tolist()


def get_results_from_predictions(
    predictions_file_path: str,
    unique_labels: List[str],
    sample_rate: int = 200,
    output_file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reads predictions from a h5 file and processes them to generate an average precision table.

    Args:
        predictions_file_path (str): h5 path to the file containing predictions. The predictions should be obtained by running the get_results_for_single_manifest.py script with the flag --export-predictions set to True.
        unique_labels (List[str]): List of the vocalization types labels used in the predictions.
        sample_rate (int): Sample rate of the targets in the predictions file.
        output_file_path (Optional[str]): Path to save the processed results. If None, returns the results as a pandas DataFrame
    Returns:
        pd.DataFrame: Processed results
    """
    results_dict = {
        "Voc. type": unique_labels,
        "Number of voc.": [],
        "Total duration voc. (min)": [],
        "Average Precision": [],
    }
    predictions, targets = get_predictions(predictions_file_path)
    _, targets_framewise = get_predictions(predictions_file_path, segmented=False)
    results_dict["Total duration voc. (min)"] = get_tot_duration_vocalizations(
        targets_framewise, sample_rate
    )

    # AP for each class
    for i in range(len(unique_labels)):
        print(unique_labels[i])
        num_elements_vox_type = np.sum(targets[:, i]).astype(int)
        results_dict["Number of voc."].append(num_elements_vox_type)
        predictions_one_class = predictions[:, i]
        targets_one_class = targets[:, i].astype(int)
        results_dict["Average Precision"].append(
            average_precision_score(targets_one_class, predictions_one_class)
        )

    # averaged mAP for all the classes
    results_dict["Voc. type"].append("all")
    results_dict["Average Precision"].append(
        average_precision_score(
            targets.flatten().astype(int), predictions.flatten(), average="micro"
        )
    )
    results_dict["Number of voc."].append(np.sum(targets))

    tot_duration = np.array(results_dict["Total duration voc. (min)"]).sum()
    if "focal" in unique_labels:
        tot_duration -= results_dict["Total duration voc. (min)"][
            unique_labels.index("focal")
        ]
    results_dict["Total duration voc. (min)"].append(tot_duration)

    results_df = pd.DataFrame(results_dict)
    results_df.sort_values(by="Average Precision", ascending=False, inplace=True)
    results_df = results_df.round(3)
    if output_file_path is not None:
        results_df.to_csv(output_file_path, index=False)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-file-path",
        type=str,
        required=True,
        help="Path to the predictions h5 file. The latter can be obtained by running the get_results_for_single_manifest.py script with the flag --export-predictions set to True.",
    )
    parser.add_argument(
        "--unique-labels",
        type=str,
        help="A string list that contains the names of the vocalization types.",
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        default=None,
        help="Path to save the results as a csv file. If not provided, the results will not be saved.",
    )
    args = parser.parse_args()
    get_results_from_predictions(
        predictions_file_path=args.predictions_file_path,
        unique_labels=ast.literal_eval(args.unique_labels),
        output_file_path=args.output_file_path,
    )
