#!/usr/bin/env python3 -u
# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing for the Meerkat data within the CCAS project.
"""
import os
import h5py
import argparse
import librosa
import string
import random
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta, datetime
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment-length",
        default=10,
        type=int,
        help="Length in seconds into which the audio files should be segmented",
    )
    parser.add_argument(
        "--resample-rate",
        default=8000,
        type=int,
        help="Sample rate to which everything is resampled",
    )
    parser.add_argument(
        "--use-only-files-with-labels",
        default=False,
        type=bool,
        help="If you have multiple audio files with and without label information and you "
             "only want to segment and prepare those files for which you "
             "have - at least partially - label information provided. Then set this to True. "
             "This is only used when a valid pickle-file is provided",
    )
    parser.add_argument(
        "--randomize-file-names",
        default=False,
        type=bool,
        help="If true, randomize the filenames of the output segments. We will save a dictionary"
             "that maps every randomized name to its original name, that would've been"
             "used if this flag were false",
    )
    parser.add_argument(
        "--base-name", default="MeerKAT",
        type=str, metavar="DIR", help="The base name for this project. "
                                      "The output folder will have this a prefix"
    )
    parser.add_argument(
        "--input-folder", default=Path.home(), type=list_of_strings,
        help="The base folder(s) for the wav files. "
             "Nested folders are supported and multiple folders can be passed "
             "as comma-separated list. For example --input-folder='path/one','path/two'"
    )
    parser.add_argument(
        "--output-folder", default=Path.home(),
        type=str, metavar="DIR", help="The destination folder. "
                                      "This folder will be created if it does not exist. "
                                      "This script will produce a single subfolder with the "
                                      "name given by {base_name}_{segment_length}s_{current_date} "
                                      "and then two subfolders: 'wav' and 'lbl'"
    )
    parser.add_argument(
        "--pickle-file", default=Path.home(),
        type=str, help="The path to the pickle file that holds label information. "
                       "This pickle file should be a Pandas DataFrame containing "
                       "Five columns: 'Name', 'AudioFile', 'StartRelative', 'EndRelative', 'Focal'. "
                       "'Name' is the name of the class. 'Song One', for example. "
                       "'AudioFile' is the name of the original Audiofile in which this event happens. "
                       "'StartRelative' is a timedelta object indicating the start time relative to the file start. "
                       "'EndRelative' is a timedelta object indicating the end time relative to the file start. "
                       "'Focal' is a boolean Flag indicating if the event is focal or not ."
    )
    parser.add_argument(
        "--stereo-file", default=Path.home(),
        type=str, help="You can provide a csv file that holds "
                       "three columns index, wavFile, meerkatChannel."
                       "If you are not providing this file the "
                       "zeroth channel will be used per default"
    )
    return parser


def iteration(file, args, id_generator, labels, channel_dict):
    # Then we load the dataframe that was created using the "raw_labels_to_dataframe.py" script
    # Create dictionary
    random_names_dictionary = {}
    out_folder = os.path.join(args.output_folder, "{}_{:02.0f}s_{}".format(
        args.base_name, args.segment_length, datetime.today().strftime('%Y-%m-%d')))

    c_ = 0
    total_duration = []
    prev_audio_filename = ""
    audio_filename = str(file)
    base_file_name = os.path.basename(audio_filename)
    if os.path.isfile(args.pickle_file) and labels is not None:
        if (base_file_name not in args.audio_file_list_from_pickle) and args.use_only_files_with_labels:
            return total_duration, c_, random_names_dictionary

        starts = labels.where(labels.AudioFile == base_file_name).dropna(how="all").StartRelative.tolist()
        ends = labels.where(labels.AudioFile == base_file_name).dropna(how="all").EndRelative.tolist()
        file_labels = labels.where(labels.AudioFile == base_file_name).dropna(how="all").Name
        file_focal_labels = labels.where(labels.AudioFile == base_file_name).dropna(how="all").Focal

    c_ += 1
    try:
        waveform, sample_rate = sf.read(audio_filename)
        metadata = sf.info(audio_filename)
    except sf.LibsndfileError as e:
        print("{} is corrupt and raised a LibsndfileError. We skip it.".format(audio_filename), flush=True)
        return total_duration, c_, random_names_dictionary
    if len(waveform) == 0:
        print("{} is corrupt and has a length of zero. We skip it.".format(audio_filename), flush=True)
        return total_duration, c_, random_names_dictionary

    if waveform.ndim == 2:  # Stereo file
        if channel_dict is not None:
            arg_channel_dict = [x[:-4] in os.path.basename(audio_filename).lower() for x in channel_dict]
            if any(arg_channel_dict):
                arg_channel_dict_keys = list(channel_dict.keys())[np.argwhere(arg_channel_dict).squeeze()]
                waveform = waveform[:, channel_dict[arg_channel_dict_keys]].squeeze()
            else:
                if audio_filename[:-18] != prev_audio_filename:
                    print(os.path.basename(audio_filename).lower()[:-18],
                          " has no entry in overview file but is stereo, choosing 0")
                    prev_audio_filename = audio_filename[:-18]
                waveform = waveform[:, 0].squeeze()  # Use the first channel if no ext info is avail.
        else:
            waveform = waveform[:, 0].squeeze()

    resample_cond = metadata.samplerate != args.resample_rate or metadata.channels != 1

    # Prepare the base filename for output
    # if files were structured in multiple nested folders, then encode
    # this into the new filename starting from the base input folder
    base_out_filename = os.path.dirname(audio_filename).replace(os.sep, "_").replace("-", "_").lower()
    if base_out_filename.startswith("_"):
        base_out_filename = base_out_filename[1:]
    # TODO: input_path is a list ... len is always 1, or 2, or so
    base_out_filename = base_out_filename[len(args.input_folder):]

    # Split the file
    # TODO: When max(waveform.shape) == segment_length_sec * sample_rate nothing happens
    segments = np.arange(0, max(waveform.shape), args.segment_length * sample_rate).astype(int)
    pargs = (base_file_name, len(segments) - 1)
    print("File {} has {} segments".format(*pargs))
    for low, high in tqdm(zip(segments[:-1], segments[1:]), leave=False):
        wave_snippet = waveform[low:high]
        # print("\nWave snippet", wave_snippet.shape)
        from_sec, to_sec = low / sample_rate, high / sample_rate

        if os.path.isfile(args.pickle_file):
            # convert to timedeltas for label checking
            from_sec_td, to_sec_td = timedelta(seconds=from_sec), timedelta(seconds=to_sec)

        if resample_cond:  # Resample if needed
            wave_snippet = librosa.resample(wave_snippet,
                                            orig_sr=sample_rate,
                                            target_sr=args.resample_rate,
                                            res_type="kaiser_best")

        f_name_base = "{}_{:05.0f}s_{:05.0f}s".format(os.path.basename(audio_filename)[:-4],
                                                      from_sec, to_sec)
        if args.randomize_file_names:
            _new_rand_name = id_generator()
            if _new_rand_name in random_names_dictionary:
                while _new_rand_name in random_names_dictionary:  # Create a new name until it is unique
                    _new_rand_name = id_generator()
            random_names_dictionary.update({_new_rand_name: f_name_base})
            f_name_base = _new_rand_name
            out_folder_wav = os.path.join(out_folder, "wav", "{:05.0f}Hz".format(args.resample_rate))
            out_folder_lbl = os.path.join(out_folder, "lbl", "{:05.0f}Hz".format(args.resample_rate))
        else:
            out_folder_wav = os.path.join(out_folder, "wav", "{:05.0f}Hz".format(args.resample_rate),
                                          base_out_filename)
            out_folder_lbl = os.path.join(out_folder, "lbl", "{:05.0f}Hz".format(args.resample_rate),
                                          base_out_filename)

        f_name_wav = f_name_base + ".wav"

        if not os.path.isdir(out_folder_wav):
            os.makedirs(out_folder_wav)

        if os.path.isfile(os.path.join(out_folder_wav, f_name_wav)):  # Avoid creating files that already exist
            print("The file {} already exists, skipping rewriting it".format(f_name_wav))
            continue

        # Write out wav file
        try:
            sf.write(os.path.join(out_folder_wav, f_name_wav),
                     wave_snippet, args.resample_rate,
                     format="WAV",
                     subtype="PCM_16")
        except sf.LibsndfileError as e:
            print("{} cannot be written out and raised a LibsndfileError. We skip it.".format(audio_filename),
                  flush=True)
            print(e, flush=True)
            continue

        if not os.path.isfile(os.path.join(out_folder_wav, f_name_wav)):  # Check that writing was succesful
            print("The file {} was not written to {}".format(f_name_wav, out_folder_wav))

        # Check for labels
        start_time_lbl, start_frame_lbl, end_time_lbl, end_frame_lbl, lbl, lbl_cat, foc = [], [], [], [], [], [], []
        if os.path.isfile(args.pickle_file) and labels is not None:
            for lbl_i, (s, e) in enumerate(zip(starts, ends)):
                # If a label is fully or partial in the time interval
                if s < from_sec_td < e or s < to_sec_td < e or from_sec_td < s < e < to_sec_td:
                    start_time = (s - from_sec_td).total_seconds()
                    start_time_lbl.append(max(start_time, 0.))
                    start_frame_lbl.append(
                        np.floor(start_time_lbl[-1] * args.resample_rate).astype(int) if start_time > 0 else 0)

                    end_time = (e - from_sec_td).total_seconds()
                    end_time_lbl.append(min(end_time, float(args.segment_length)))
                    end_frame_lbl.append(np.ceil(end_time_lbl[-1] * args.resample_rate).astype(int))

                    lbl.append(file_labels.iloc[lbl_i])
                    foc.append(1 if file_focal_labels.iloc[lbl_i].lower() == "focal" else 0)

                    lbl_cat.append(args.unique_labels.index(lbl[-1]))
                    total_duration.append(end_time_lbl[-1] - start_time_lbl[-1])

        if not os.path.isdir(out_folder_lbl):
            os.makedirs(out_folder_lbl)

        f_name_lbl = f_name_base + ".h5"
        with h5py.File(os.path.join(out_folder_lbl, f_name_lbl), mode="w") as f:
            f.create_dataset(name="start_time_lbl", data=start_time_lbl)
            f.create_dataset(name="start_frame_lbl", data=start_frame_lbl)
            f.create_dataset(name="end_time_lbl", data=end_time_lbl)
            f.create_dataset(name="end_frame_lbl", data=end_frame_lbl)
            f.create_dataset(name="lbl", data=lbl)
            f.create_dataset(name="lbl_cat", data=lbl_cat)
            f.create_dataset(name="foc", data=foc)
    return total_duration, c_, random_names_dictionary


def main(args):
    if args.randomize_file_names:
        # Create randomizer function
        def id_generator(size=32, chars=string.ascii_letters + string.digits):
            # For a 32-character string, there are 62^32 (approximately 2.28 x 10^57)
            # possible combinations. The probability of generating a specific string
            # is 1 / (62^32). After generating one string, the probability that the
            # second string matches the first is also 1 / (62^32).
            # Not gonna happen.
            return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    else:
        id_generator = None

    supported_file_types = ["WAV", "AIFF", "AIFC", "FLAC", "OGG", "MP3", "MAT"]
    files = []
    for d in args.input_folder:
        for ft in supported_file_types:
            regex_ft = "".join(["[{}{}]".format(x.lower(), x.upper()) for x in ft])
            files += list(Path(d).rglob('*.{}'.format(regex_ft)))

    if os.path.isfile(args.pickle_file):
        try:
            labels = pd.read_pickle(args.pickle_file)
            args.audio_file_list_from_pickle = labels.AudioFile.to_list()
            args.unique_labels = list(labels.Name.unique())
        except Exception as e:
            labels = None
            print("Pickle file not found or could not be read. If you want to use labels"
                  "for a finetuning, you have to provide a valid file. Here is the raised error.")
            print(e)
            print("\nThis script continues without using the file.\n")
    else:
        labels = None

    # Read in channel dictionary (contains info which channel to use in stereo files)
    if os.path.isfile(args.stereo_file):
        try:
            channel_tab = pd.read_csv(args.stereo_file, sep=",")
            channel_dict = {x.lower(): y for x, y in zip(channel_tab.wavFile, channel_tab.meerkatChannel)}
        except Exception as e:
            channel_dict = None
            print("Stereo file not found or could not be read. If you want to use your own list"
                  "with information on what stereo channel to use, you have"
                  "to provide a valid file. Use a comma-separated csv file with a header of the form:"
                  "index, wavFile, meerkatChannel. Here is the raised error.\n")
            print(e)
            print("\nThis script continues without using the file.\n")
    else:
        channel_dict = None

    # Remove duplicates
    files = list(set(files))
    num_threads = min(int(multiprocessing.cpu_count() / 2), len(files))

    # Using ThreadPoolExecutor
    partial_iteration = partial(iteration, args=args, id_generator=id_generator,
                                labels=labels, channel_dict=channel_dict)

    with tqdm(total=len(files)) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(partial_iteration, data) for data in files]

            results = []
            for future in as_completed(futures):
                result = future.result()  # Get the result (optional)
                results.append(result)
                pbar.update(1)  # Update the progress bar

    total_duration = sum([x[0] for x in results])
    c_ = sum([x[1] for x in results])
    random_names_dictionary = [x[2] for x in results]
    random_names_dictionary = {k: v for d in random_names_dictionary for k, v in d.items()}

    print("We iterated over {} files".format(c_), flush=True)
    if args.randomize_file_names:
        # write out dictionary
        csv_out = os.path.join(args.output_folder, "{}_{:02.0f}s_{}_randomized_dictionary.csv".format(
            args.base_name, args.segment_length, datetime.today().strftime('%Y-%m-%d')))
        header = ["RandomizedName", "OriginalName"]
        with open(csv_out, "w") as f:
            f.write("\t".join(header) + "\n")  # write header -> tab separated
            for k, v in random_names_dictionary.items():
                f.write("\t".join([k, v]) + "\n")  # write dictionary -> tab separated

    print("Total duration of all calls in all files: {:02.02f}s".format(np.sum(total_duration)), flush=True)
    print("Total number of all files: {:02.00f}".format(len(total_duration)), flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
