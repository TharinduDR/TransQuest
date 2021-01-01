[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/transquest)](https://pepy.tech/project/transquest)

# TransQuest : Translation Quality Estimation with Cross-lingual Transformers. 

TransQuest provides state-of-the-art models for translation quality estimation.

We are the winning solution in WMT 2020 Quality Estimation Shared Task -  Sentence-Level Direct Assessment.

Official Documentation is available on - https://tharindudr.github.io/TransQuest.

### Features
- Sentence-level translation quality estimation on both aspects: predicting post editing efforts and direct assessment.
- Perform significantly better than current state-of-the-art quality estimation methods like DeepQuest and OpenKiwi in all the languages experimented. 
- Pre-trained quality estimation models for fifteen language pairs.  

## Installation
You first need to install PyTorch. The recommended PyTorch version is 1.5.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install TransQuest from source or from pip. 

#### From pip

```bash
pip install transquest
```

#### From Source
```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```

Steps that you need to follow after installing TransQuest is available here - https://tharindudr.github.io/TransQuest/architectures/.

## Run the examples
We have included several examples on how to use TransQuest on recent WMT QE shared tasks. 

1. [WMT 2020 Sentence-level Direct Assessment QE Shared Task](examples/sentence_level/wmt_2020)
2. [WMT 2020 Sentence-level Post-Editing Effort QE Shared Task](examples/sentence_level/wmt_2020_task2)
3. [WMT 2019 Sentence-level Post-Editing Effort QE Shared Task](examples/sentence_level/wmt_2019)
3. [WMT 2018 Sentence-level Post-Editing Effort QE Shared Task](examples/sentence_level/wmt_2018)

More details of the examples can be found in https://tharindudr.github.io/TransQuest/examples/.

## TransQuest Model Zoo
We released several pre-trained models. You will be able to find them here - https://tharindudr.github.io/TransQuest/pretrained/. We will be keep releasing new pre-trained model. Please keep in touch.

## Citations
Please consider citing us if you use the library. 
```bash
@InProceedings{transquest:2020,
author = {Ranasinghe, Tharindu and Orasan, Constantin and Mitkov, Ruslan},
title = {TransQuest: Translation Quality Estimation with Cross-lingual Transformers},
booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
year = {2020}
}
```

The task specific paper on 2020 WMT sentence-level DA that won the first place in the competition. 
```bash
@InProceedings{transquest:2020,
author = {Ranasinghe, Tharindu and Orasan, Constantin and Mitkov, Ruslan},
title = {TransQuest at WMT2020: Sentence-Level Direct Assessment},
booktitle = {Proceedings of the Fifth Conference on Machine Translation},
year = {2020}
}
```