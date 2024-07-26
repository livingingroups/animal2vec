import os
import torch
import h5py
import argparse
import contextlib
import numpy as np
from tqdm import tqdm
from itertools import groupby
from pathlib import Path

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

from torch.utils.data import DataLoader
from fairseq import checkpoint_utils
from nn.audio_tasks import FileAudioLabelDataset  # for registering everything


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize",
        default=True,
        type=bool,
        help="Should we normalize the input to zero mean and unit variance",
    )
    parser.add_argument(
        "--export_embeddings",
        default=False,
        type=bool,
        help="Should we export the embeddings (This will produce a very large file)",
    )
    parser.add_argument(
        "--export_predictions",
        default=False,
        type=bool,
        help="Should we export the predictions",
    )
    parser.add_argument(
        "--use_softmax",
        default=False,
        type=bool,
        help="If set to True, then we use a softmax as the final activation."
             "Otherwise, we use a sigmoid.",
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(str(Path.home()), "eas_shared", "ccas", "working", "animal2vec",
                             "38ms_receptive_sinc_ft_full-0_825-4_asmodeus",
                             "checkpoints", "checkpoint_last.pt"),
        type=str,
        help="Path to pretrained model. "
             "This should point to a *.pt file created by pytorch."
    )
    parser.add_argument(
        "--device", default="cuda",
        type=str, choices=["cuda", "cpu"],
        help="The device you want to use. "
             "We will fall back to cpu if no cuda device is available."
    )
    parser.add_argument(
        "--batch_size", default=12,
        type=int,
        help="Every file that is being split into smaller segments and fed to the model"
             "in batches. --batch-size gives this batch size and --segment-length gives"
             "the segment length in sec. Default values are 10s segment length and "
             "12 segments batch size."
             "Adjust if you have more - or less. Keep in mind that first reducing --batch-size"
             "is recommend as reducing segment-length reduces the timespan the transformer"
             "can use for contextualizing."
    )
    parser.add_argument(
        "--average_start_k_layers", default=0,
        type=int,
        help="The transformer layer from which we start averaging"
    )
    parser.add_argument(
        "--average_end_k_layers", default=16,
        type=int,
        help="The transformer layer with which we end averaging"
    )
    parser.add_argument(
        "--out_path", default=os.path.curdir,
        type=str,
        help="The path to where embeddings.h5 file should be written to."
             "The folder does not need to exists. The script can take care of that. "
             "Default is the current workdir."
    )
    parser.add_argument(
        "--conv_feature_layers",
        default='[(127, 63, 1)] +[(512, 10, 5)] + [(512, 3, 2)] * 3 + [(512, 3, 1)] + [(512, 2, 1)] * 2',
        type=str,
        help="string describing convolutional feature extraction layers in form of a python list that contains "
             "[(dim, kernel_size, stride), ...]"
    )
    parser.add_argument(
        "--unique_labels",
        default="['beep', 'synch', 'sn', 'cc', 'ld', 'oth', 'mo', 'al', 'soc', 'agg', 'eating', 'focal']",
        type=str,
        help="A string list that, when evaluated, holds the names for the used classes."
    )
    parser.add_argument(
        "--manifest_path", default=os.path.curdir,
        type=str,
        help="The path to the manifest file folder."
    )
    parser.add_argument(
        "--split", default="valid_0",
        type=str,
        help="The split manifest file to load."
    )
    parser.add_argument(
        "--method",
        default="avg",
        type=str,
        choices=["avg", "max", "canny"],
        help="Which method to use for fusing the predictions into time bins."
             "avg: Average pooling, then thresholding."
             "max: Max pooling, then thresholding."
             "canny: Canny edge detector",
    )
    parser.add_argument(
        "--sigma_s",
        default=0.1,
        type=float,
        help="Size of Gaussian (std dev) in seconds for the canny method. "
             "Filter width in seconds for avg and max methods.",
    )
    parser.add_argument(
        "--metric_threshold",
        default=0.125,
        type=float,
        help="Threshold for the filtered predictions. Only used when --method={max, avg}",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.,
        type=float,
        help="The minimum IoU that is needed for a prediction to be counted as "
             "overlapping with focal. 0 means that even a single frame overlap is enough, "
             "while 1 means that only a perfect overlap is counted as overlap.",
    )
    parser.add_argument(
        "--maxfilt_s",
        default=0.1,
        type=float,
        help="Time to smooth with maxixmum filter. Only used when --method=canny",
    )
    parser.add_argument(
        "--max_duration_s",
        default=0.5,
        type=float,
        help="Detections are never longer than max duration (s). Only used when --method=canny",
    )
    parser.add_argument(
        "--sample_rate",
        default=8000,
        type=int,
        help="The sample rate in Hz for the input data",
    )
    parser.add_argument(
        "--min_label_size",
        default=3032,
        type=int,
        help="if set, we only load files from the manifest whose label file bytesize "
             "is larger then this value. Hint: Empty label files have Bytesize 3032."
             "Only used when with_labels is True",
    )
    parser.add_argument(
        "--lowP",
        default=0.125,
        type=float,
        help="Low threshold. Detections are based on Canny edge detector, "
             "but the detector sometimes misses the minima on some transients"
             "if they are not sharp enough.  lowP is used for pruning detections."
             "Detections whose Gaussian-smoothed signal is beneath lowP are discarded."
             "Only used when --method=canny",
    )
    return parser


def get_intervalls(data, shift=0):
    # Group the array in segments with itertools groupby function
    grouped = (list(g) for _, g in groupby(enumerate(data), lambda t: t[1]))
    # Only add the interval if it is with values larger than 0
    return [(g[0][0] + shift, min([len(data) - 1, g[-1][0] + shift])) for g in grouped if g[0][1] == 1]


def main(args):
    model_path = args.model_path
    assert os.path.isfile(model_path)

    # Check if out path exists
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    method_dict = {
        "sigma_s": args.sigma_s,
        "metric_threshold": args.metric_threshold,
        "maxfilt_s": args.maxfilt_s,
        "max_duration_s": args.max_duration_s,
        "lowP": args.lowP,
        "iou_threshold": args.iou_threshold,
    }

    # load the model
    device = "cuda" if torch.cuda.is_available() and args.device.lower() == "cuda" else "cpu"
    print("Loading the model and placing on {} ... ".format(device), end="")
    models, cfg = checkpoint_utils.load_model_ensemble([model_path])
    model = models[0].to(device)  # place on appropriate device
    print("done")

    # The labels on which the model was trained on
    # print("\n args.unique_labels", args.unique_labels)
    print("Loading the data ... ", end="")
    dataset = FileAudioLabelDataset(
        manifest_path=os.path.join(args.manifest_path, args.split + ".tsv"),
        sample_rate=args.sample_rate,
        min_sample_size=1612,
        normalize=args.normalize,
        shuffle=False,
        return_labels=True,
        unique_labels=eval(args.unique_labels),
        use_focal_loss=cfg.criterion.use_focal_loss,
        min_label_size=args.min_label_size,
        conv_feature_layers=args.conv_feature_layers,
        segmentation_metrics=False,
    )
    # args.export_embeddings = False
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    print("done")

    # file name will "embeddings_model-name_checkpoint-name_avg-start_avg-end_split-name_foldername.h5"
    folder_name = model_path[len(model_path[:model_path.find("/checkpoints/")]) -
                             model_path[
                             :model_path.find("/checkpoints/")][::-1].find("/"):model_path.find("/checkpoints/")]
    file_args = (model.__class__.__name__, os.path.basename(model_path),
                 args.average_start_k_layers, args.average_end_k_layers,
                 args.split, folder_name)
    if args.export_embeddings:
        emb_out_filename = "embeddings_{}_{}_{}_{}_{}_{}.h5".format(*file_args)
        out_path = os.path.join(args.out_path, emb_out_filename)
        if os.path.isfile(out_path):
            os.remove(out_path)
        embedding_context = h5py.File(out_path, "w")
    else:
        embedding_context = contextlib.nullcontext()

    if args.export_predictions:
        pred_out_filename = "predictions_{}_{}_{}_{}_{}_{}.h5".format(*file_args)
        out_path = os.path.join(args.out_path, pred_out_filename)
        if os.path.isfile(out_path):
            os.remove(out_path)
        prediction_context = h5py.File(out_path, "w")
    else:
        prediction_context = contextlib.nullcontext()

    if args.export_predictions or args.export_embeddings:
        with torch.inference_mode(), embedding_context as f_h5, prediction_context as f_pred_h5:
            model.eval()
            for samples in tqdm(dataloader,
                                desc="Starting inference"):
                # print(targets.size())
                source = samples["source"].squeeze()
                # print(source.size())
                res = model.extract_features(source=source.to(device))
                targets = samples["target"]

                if cfg.criterion.use_focal_loss:
                    seg_target_idx = [[get_intervalls(x) for x in y.T] for y in targets]
                else:
                    seg_target_idx = [get_intervalls(y) for y in targets]

                samples["seg_target_idx"] = seg_target_idx
                samples["source_size"] = source.size(-1)

                if args.export_predictions:
                    if "linear_eval_projection" in res:
                        if args.use_softmax:
                            probs = torch.softmax(res["linear_eval_projection"].float(), -1)
                        else:
                            probs = torch.sigmoid(res["linear_eval_projection"].float())
                    else:
                        if args.use_softmax:
                            probs = torch.softmax(res["encoder_out"].float(), -1)
                        else:
                            probs = torch.sigmoid(res["encoder_out"].float())
                    if not args.use_softmax:
                        pr, ta, ios, sp, me = model.get_segmented_probs_and_targets(
                            samples,
                            torch.tensor(probs), method_dict,
                            method=args.method
                        )

                    bs = probs.size(0)
                    time_dim = probs.size(1)
                    likelihoods_export = probs.detach().cpu().numpy()
                    targets_export = targets.detach().cpu().numpy()
                    if not args.use_softmax:
                        segmented_likelihoods_export = pr.detach().cpu().numpy()
                        segmented_targets_export = ta.detach().cpu().numpy()
                    else:
                        segmented_likelihoods_export = likelihoods_export.copy()
                        segmented_targets_export = targets_export.copy()
                    segmented_likelihoods_export = segmented_likelihoods_export.reshape(bs, time_dim, -1)
                    segmented_targets_export = segmented_targets_export.reshape(bs, time_dim, -1)

                    like_tar_shapes = likelihoods_export.shape == targets_export.shape
                    seg_like_seg_tar_shapes = segmented_likelihoods_export.shape == segmented_targets_export.shape
                    seg_like_like_shapes = segmented_likelihoods_export.shape == likelihoods_export.shape
                    seg_tar_tar_shapes = segmented_targets_export.shape == targets_export.shape
                    _assert(
                        like_tar_shapes and seg_like_seg_tar_shapes and seg_like_like_shapes and seg_tar_tar_shapes,
                        "Predictions and Target do not share the first two dimensions.\n"
                        "Model Name: {},"
                        "Predictions Shape: {}, Target Shape {},"
                        "Segmented Predictions Shape: {}, Segmented Target Shape {}".format(
                            model.__class__.__name__,
                            likelihoods_export.shape, targets_export.shape,
                            segmented_likelihoods_export.shape, segmented_targets_export.shape
                        )
                    )

                    for enu, (like, seg_like, seg_tar, tar) in enumerate(zip(likelihoods_export,
                                                                             segmented_likelihoods_export,
                                                                             segmented_targets_export,
                                                                             targets_export)):  # Iterate over Batch
                        # for enu, (like, tar) in enumerate(zip(likelihoods_export,
                        #                                       targets_export)):  # Iterate over Batch
                        # Create group with unix index
                        index = samples["id"][enu]
                        grp_pred = f_pred_h5.create_group("{:06.0f}".format(index))
                        # This group holds information about the filename, the target, and the embedding
                        fn_pred = dataset.fnames[index]
                        fn_pred = fn_pred if isinstance(dataset.fnames, list) else fn_pred.as_py()
                        fn_pred = dataset.text_compressor.decompress(fn_pred)
                        grp_pred.create_dataset("fname", data=fn_pred)
                        grp_pred.create_dataset("likelihood", data=like, dtype=np.float32)  # BTC
                        if not args.use_softmax:
                            grp_pred.create_dataset("segmented_likelihood", data=seg_like, dtype=np.float32)  # BTC
                            grp_pred.create_dataset("segmented_target", data=seg_tar, dtype=np.float32)  # BTC
                        grp_pred.create_dataset("target", data=tar, dtype=np.float32)  # BTC

                if args.export_embeddings:
                    layer_results = res["layer_results"]
                    min_layer = args.average_start_k_layers
                    max_layer = args.average_end_k_layers

                    finetuned = False
                    if hasattr(model, "w2v_encoder"):  # is finetuned or not
                        finetuned = True

                    d2v = getattr(model, "is_d2v_multi", False) or "data2vec" in model.__class__.__name__.lower()
                    if finetuned:
                        if hasattr(model.w2v_encoder, "is_d2v_multi"):  # finetuned data2vec model
                            d2v = getattr(model.w2v_encoder, "is_d2v_multi", False)

                    if d2v:
                        target_layer_results = [l for l in layer_results[min_layer:max_layer]]
                    else:
                        target_layer_results = [l[0] for l in layer_results[min_layer:max_layer]]
                    if not d2v:  # transpose if w2v
                        target_layer_results = [tl.transpose(0, 1) for tl in target_layer_results]  # HTBC -> HBTC
                    # Average over transformer layers
                    avg_emb = (sum(target_layer_results) / len(target_layer_results)).float()  # BTC

                    embeddings_export = avg_emb.detach().cpu().numpy()
                    targets_export = targets.detach().cpu().numpy()
                    _assert(
                        embeddings_export.shape[:2] == targets_export.shape[:2],
                        "Embeddings and Target do not share the first two dimensions.\n"
                        "Model Name: {}, d2v detected: {}\n"
                        "Embeddings Shape: {}, Target Shape {}".format(
                            model.__class__.__name__, d2v,
                            embeddings_export.shape, targets_export.shape)
                    )

                    for enu, (emb, tar) in enumerate(zip(embeddings_export, targets_export)):  # Iterate over Batch
                        # Create group with unix index
                        index = samples["id"][enu]
                        grp = f_h5.create_group("{:06.0f}".format(index))
                        # This group holds information about the filename, the target, and the embedding
                        fn = dataset.fnames[index]
                        fn = fn if isinstance(dataset.fnames, list) else fn.as_py()
                        fn = dataset.text_compressor.decompress(fn)
                        grp.create_dataset("fname", data=fn)
                        grp.create_dataset("embedding", data=emb, dtype=np.float32)  # BTC
                        grp.create_dataset("target", data=tar, dtype=np.float32)  # BTC


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    main(args)
