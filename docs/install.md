# Installation
You first need to install PyTorch. The recommended PyTorch version is 1.8.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install TransQuest from source or from pip.
If you are training models, we highly recommend using a GPU. We used a NVIDIA TESLA K80 GPU to train the models.

### From pip

```bash
pip install transquest
```

### From Source

```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```

Now that you have installed TransQuest, it is time to check our architectures in [sentence-level](https://tharindudr.github.io/TransQuest/architectures/sentence_level_architectures/) and [word-level.](https://tharindudr.github.io/TransQuest/architectures/word_level_architecture/)




