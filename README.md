## animal2vec: A self-supervised transformer for rare-event raw audio input

This is the official repository for [animal2vec](https://arxiv.org/abs/2406.01253#):

```
Schäfer-Zimmermann, J. C., Demartsev, V., Averly, B., Dhanjal-Adams, K., Duteil, M., Gall, G., Faiss, M., Johnson-Ulrich, L., Stowell, D., Manser, M., Roch, M. A. & Strandburg-Peshkin, A. (2024)
animal2vec and MeerKAT: A self-supervised transformer for rare-event raw audio input and a large-scale reference dataset for bioacoustics
arXiv preprint arXiv:2406.01253 (2024)
```

### Paper abstract
Bioacoustic research, vital for understanding animal behavior, conservation, and ecology, faces a monumental challenge: analyzing vast datasets where animal vocalizations are rare. While deep learning techniques are becoming standard, adapting them to bioacoustics remains difficult. We address this with animal2vec, an interpretable large transformer model, and a self-supervised training scheme tailored for sparse and unbalanced bioacoustic data. It learns from unlabeled audio and then refines its understanding with labeled data. Furthermore, we introduce and publicly release MeerKAT: **Meer**kat **K**alahari **A**udio **T**ranscripts, a dataset of meerkat (Suricata suricatta) vocalizations with millisecond-resolution annotations, the largest labeled dataset on non-human terrestrial mammals currently available. Our model outperforms existing methods on MeerKAT and the publicly available NIPS4Bplus birdsong dataset. Moreover, animal2vec performs well even with limited labeled data (few-shot learning). animal2vec and MeerKAT provide a new reference point for bioacoustic research, enabling scientists to analyze large amounts of data even with scarce ground truth information.


### MeerKAT dataset
You can find the publicly available MeerKAT dataset at the Max Planck data repository Edmond using [this link](https://doi.org/10.17617/3.0J0DYB).

The majority of the audio originates from acoustic collars (Edic Mini Tiny+ A77, Zelenograd, Russia, which sample at 8kHz with 10bit quantization) that were attached to the animals (41 individuals throughout two campaigns in 2017 and 2019), where each file corresponds to a recording for a single individual and day. The remainder of the dataset was recorded using Marantz PMD661 digital recorders (Carlsbad, CA, U.S.) attached to directional Sennheiser ME66 microphones (Wedemark, Germany) sampling at 48kHz with 32bit quantization. When recording, field researchers held the microphones close to the animals (within 1m). The data were recorded during times when meerkats typically forage for food by digging in the ground for small prey. See our paper and [1] and [2] for more details.

MeerKAT is released as 384 592 10-second samples, amounting to 1068 h, where 66 398 10-second samples (184 h) are labeled and ground-truth-complete; all call and recurring anthropogenic events in this 184 h are labeled. For further details, see [2]. All samples have been standardized to a sample rate of 8 kHz with 16-bit quantization, sufficient to capture the majority of meerkat vocalization frequencies (the first two formants are below the Nyquist frequency of 4 kHz [46]). The total dataset size of 59 GB (61 GB, including the label files) is comparatively small, making MeerKAT easily accessible and portable despite its extensive length. Each 10-second file has an accompanying HDF5 label file that lists label categories, start and end time offsets (s), and a "focal" designation indicating whether the call was given by the collar-wearing or followed individual or not.

By agreement with the Kalahari Research Centre (KRC), we have made these data available in a way that can further machine learning research without compromising the ability of the KRC to continue conducting valuable ecological research on these data.

Consequently, the filenames of the 10-second samples have been randomly sampled, and their temporal order and individual identity cannot be recovered, but can be requested from us.

### Installation requirements for animal2vec
We strongly recommend to use either [Python venv](https://docs.python.org/3/library/venv.html) or [Anaconda](https://anaconda.cloud/getting-started-with-conda-environments) to create a virtual python environment.<br>
Tested python versions are 3.6 to 3.9. Note, **3.10** will not work due to a complicated module conflict.

Then do:
``` bash
pip install --upgrade pip==24.0 # pip version higher than that won't install the needed omegaconf version

# install the needed modules from the requirements file
pip install -r /path/to/this/repo/requirements.txt

# update fairseq directly from the repo
pip install git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f
```
We are working on providing a docker image and will update this repo once it becomes available.

### MeerKAT-trained animal2vec model weights
You can find the weights for our MeerKAT-pretrained and finetuned animal2vec models at the Max Planck data repository Edmond using [this link](https://doi.org/10.17617/3.ETPUKU).<br>
There, you find the [fairseq](https://github.com/facebookresearch/fairseq) model checkpoints (the file ending is *.pt*). You can load in these weights using the [fairseq](https://github.com/facebookresearch/fairseq) framework.

For example like this:<br>
**You need to be in the root directory of this repo for this code snippet to work**
``` python
# The 'nn' import only works if you are in the root directory of the a2v repo, 
# This is always needed to register the model and tasks objects. Otherwise, the
# fairseq routines will through an error using our models
import nn
import torch
import numpy as np
from fairseq import checkpoint_utils

# These are the class names in the MeerKAT dataset
meerkat_class_names = ['beep', 'synch', 'sn', 'cc', 'ld', 'oth', 'mo', 'al', 'soc', 'agg', 'eating', 'focal']
path_to_pt_file = "animal2vec_large_finetuned_MeerKAT_240507.pt"
# load the model
print("\n Loading model ... ", end="")
models, model_args = checkpoint_utils.load_model_ensemble([path_to_pt_file])
print("done")

print("Moving model to cpu ... ", end="")
model = models[0].to("cpu")  # place on appropriate device
print("done\n")

# Expected shape is Batch x Time. This simulates a 10s segment at 8kHz sample rate
dummy_data = torch.rand(1, 80000).cpu()

# Generally, you should always normalize your input to zero mean and unit variance
# This repository has a helper function for that
dummy_data = nn.chunk_and_normalize(
    dummy_data,
    segment_length=10,  # Length in seconds
    sample_rate=8000,  # Sample rate of your data
    normalize=True,
    max_batch_size=16  # The max batch size your GPU or RAM can handle (16 should be ok)
)

processed_chunks = []
method_dict = {
        "sigma_s": 0.1,  # Filter width in seconds
        "metric_threshold": 0.15,  # Likelihood threshold for the predictions
        "maxfilt_s": 0.1,  # Time to smooth with maxixmum filter. Only used when method=canny
        "max_duration_s": 0.5,  # Detections are never longer than max duration (s). Only used when method=canny
        "lowP": 0.125,  # Lower likelihood threshold. Only used when method=canny
    }
with torch.inference_mode():
    model.eval()
    for bi, single_chunk in enumerate(dummy_data):
        # lists should be stacked and 1-dimensional data should be extended to 2d.
        if not torch.is_tensor(single_chunk):
            single_chunk = torch.stack(single_chunk)  # stack the list of tensors
        elif single_chunk.dim() == 1:  # split segments or single segment
            single_chunk = single_chunk.view(1, -1)
        # 1) Get frame_wise predictions
        
        # This returns a dictionary with keys: ['encoder_out', 'padding_mask', 'layer_results', 'target']
        # encoder_out is the classifier logits output (use torch.sigmoid to turn into probs)
        # padding_mask is the used padding mask (usually no padding is used, then, padding_mask is None)
        # layer_results is a list that holds the embeddings from all transformer layers
        # target is the ground truth (if provided, usually this is None, as we are not training anymore)
        net_output = model(source=single_chunk.to("cpu"))
        
        # 1.1) Convert to probalities. This has shape Batch x Time x Class (1, 2000, 12 in this example)
        probs = torch.sigmoid(net_output["encoder_out"].float())
        
        # 2) Get onset / offset predictions
        # This function calculates onset and offset and the average likelihood between the found
        # boundaries. It returns a list with len 3 with onset/offset info in seconds, their indexes,
        # and the likelihood for that segment, for every class
        fused_preds = model.fuse_predict(single_chunk.size(-1), probs,
                                        # A dictionary with information on how to estimate the onset / offset
                                         method_dict=method_dict,
                                         # Which method to use for fusing the predictions into time bins
                                         method="avg",
                                         multiplier=bi,
                                         bs=16)
        processed_chunks.append(fused_preds)

        print("We iterate over {} chunks".format(len(processed_chunks)))
        for ci, single_chunk in enumerate(processed_chunks):  # iterate over all chunks
            time_interval_batches = single_chunk[0]  # time in seconds
            likelihoods_batches = single_chunk[2]  # likelihood between 0 and 1
            
            # iterate over the segments in each chunk 
            print("\tChunk {} has {} segments".format(ci, len(time_interval_batches)))
            for t_batch, l_batch in zip(time_interval_batches, likelihoods_batches):
                # iterate over the class predictions in each batch in each chunk 
                for si, (t_seg, l_seg, n_) in enumerate(zip(t_batch, l_batch, meerkat_class_names)):
                    print("\t\tResults for Class {}: {}".format(si, n_))
                    print("\t\t\tClass {} has {} found segments.".format(n_, len(t_seg)))
                    for class_pred_time, class_pred_like in zip(t_seg, l_seg):
                        pr_args = (class_pred_time[0].numpy(), class_pred_time[1].numpy(), class_pred_like.numpy())
                        print("\t\t\t\tFrom {:02.02f}s to {:02.02f}s with a likelihood of {:02.02f}".format(*pr_args))

# With this simple example, we get an output of this structure:
#   We iterate over 1 chunks
#   	Chunk 0 has 1 segments
#   		Results for Class 0: beep
#   			Class beep has 0 found segments.
#   		Results for Class 1: synch
#   			Class synch has 1 found segments.
#   				From 0.05s to 0.32s with a likelihood of 0.24
#   		Results for Class 2: sn
#   			Class sn has 6 found segments.
#   				From 0.05s to 0.07s with a likelihood of 0.14
#   				From 0.11s to 0.19s with a likelihood of 0.16
#   				From 2.00s to 2.09s with a likelihood of 0.19
#   				From 2.48s to 2.55s with a likelihood of 0.18
#   				From 2.73s to 2.80s with a likelihood of 0.18
#   				From 4.07s to 4.16s with a likelihood of 0.19
#   		Results for Class 3: cc
#   			Class cc has 0 found segments.
#   		Results for Class 4: ld
#   			Class ld has 0 found segments.
#   		Results for Class 5: oth
#   			Class oth has 0 found segments.
#   		Results for Class 6: mo
#   			Class mo has 0 found segments.
#   		Results for Class 7: al
#   			Class al has 0 found segments.
#   		Results for Class 8: soc
#   			Class soc has 0 found segments.
#   		Results for Class 9: agg
#   			Class agg has 0 found segments.
#   		Results for Class 10: eating
#   			Class eating has 0 found segments.
#   		Results for Class 11: focal
#   			Class focal has 1 found segments.
#   				From 0.05s to 0.29s with a likelihood of 0.23

# Please note, the actual found segments will vary, as the input was created using random numbers.
```
We are working on providing model weights that can be used with the [HuggingFace Transformers module](https://huggingface.co/docs/transformers/en/index) and will update this repo once this becomes available.

### Pretrain with your own data
Coming very soon

### Finetune the MeerKAT-pretrained animal2vec model with your own data
Coming very soon

### Use the finetuned animal2vec model to do inference on your own data
Coming very soon

### Things to consider when using your own data
Coming very soon

### References

[1] Demartsev, V. et al. Signalling in groups: New tools for the integration of animal communication and collective movement. Methods Ecol. Evol. (2022).<br>
[2] Demartsev, V. et al. Mapping vocal interactions in space and time differentiates signal broadcast versus signal exchange in meerkat groups. Philos. Trans. R. Soc. Lond. B Biol. Sci. 379 (2024)
