## animal2vec and MeerKAT

This is the official repository for:

```
Schäfer-Zimmermann, Julian C.; Demartsev, Vlad; Averly, Baptiste; Dhanjal-Adams, Kiran; Duteil, Mathieu; Gall, Gabriella; Johnson-Ulrich, Lily; Stowell, Dan; Manser, Marta B.; Roch, Marie A.; Strandburg-Peshkin, Ariana
animal2vec and MeerKAT: A self-supervised transformer for rare-event raw audio input and a large-scale reference dataset for bioacoustics
```

### Install Dependencies

- Clone the repo self-supervised-animal-vocalizations and install dependencies:
    - Clone the repo:
    - ```git clone https://github.com/livingingroups/animal2vec```
    - Create a conda environment:
        - ```conda create --name ML_inference```
        - ```conda activate ML_inference```
        - ```conda install pip```
- Now switch to the repo directory and install dependencies
    - ```cd animal2vec ```
    - ```pip install -r requirements.txt```

Wait until that completes.


Then some manual installation is needed:
- Clone the fairseq repo into some directory other than our repo directory:
    - ```cd ~```
    - ```git clone https://github.com/pytorch/fairseq```
    - ```cd fairseq```
    - ```pip install --editable ./```


- **Only if you get an libcublasLt.so.11 error** You might need to uninstall a cuda library that was installed during the pytorch install, but conflicts with your active cuda driver set. See [here](https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no).
    - ```pip uninstall nvidia_cublas_cu11```

**Now you are good to go and you can use the repo.**

### Use the inference script
- Navigate to the directory where you cloned the repo (self-supervised-animal-vocalizations).
    - Check the options:
        - ```python animal2vec_inference.py --help```
    - Usually, you only need to care about the path and the out_path flag. The first is the path to a wav file or a folder of wav files. The second one is the folder into which the results are being written.
- Run the script:
    - ```python animal2vec_inference.py --path=path_to_wav_file_or_folder --out-path=path_to_results_folder --device=cpu```
