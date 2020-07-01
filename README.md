[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/transquest)](https://pepy.tech/project/transquest)

# TransQuest : Translation Quality Estimation with Cross-lingual Transformers. 

TransQuest provides state-of-the-art models for translation quality estimation.

### Features
- Sentence-level translation quality estimation on both aspects: predicting post editing efforts and direct assessment.
- Perform significantly better than current state-of-the-art quality estimation methods like DeepQuest and OpenKiwi in all the languages experimented. 
- Pre-trained quality estimation models for seven languages.  

## Installation
You first need to install PyTorch. The recommended PyTorch version is 1.5.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install TransQuest from source or from pip. 

#### From Source

```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```
#### From pip

```bash
pip install transquest
```

## Run the examples
Examples are included in the repository but are not shipped with the library.

1. [WMT 2020 Sentence-level Direct Assessment QE Shared Task](examples/wmt_2020)
2. [WMT 2020 Sentence-level Post-Editing Effort QE Shared Task](examples/wmt_2020_task2)
3. [WMT 2019 Sentence-level Post-Editing Effort QE Shared Task](examples/wmt_2019)
3. [WMT 2018 Sentence-level Post-Editing Effort QE Shared Task](examples/wmt_2018)

## TransQuest Model Zoo
Following pre-trained models are released. We will be keep releasing new models. Please keep in touch. 

| Language Pair           |  Objective |     Algorithm       |  Model Link                          | Data                                                                 | Pearson | MAE     | RMSE    |
|:-----------------------:|----------- |:-------------------:|:------------------------------------:|:--------------------------------------------------------------------:| ------: | ------: | ------: |  
| Romanian-English  (NMT) |  Direct    | TransQuest          | [model.zip](https://bit.ly/2AfuXwb)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.8982 | 0.3121  |  0.4097 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/37vT4mt)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.8501 | 0.3637  |  0.4932 |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6845 | 0.7596  |  1.0522 |
| Estonian-English (NMT)  |  Direct    | TransQuest          | [model.zip](https://bit.ly/2YjXIAa)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.7748 | 0.5904  |  0.7321 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/30mO5mW)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6804 | 0.7047  |  0.9022 |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.4770 | 0.9176  |  1.1382 |
| Nepalese-English (NMT)  |  Direct    | TransQuest          | [model.zip](https://bit.ly/2MHnCZc)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.7914 | 0.3975  |  0.5078 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/3h674bc)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6081 | 0.6531  |  0.7950 |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.3860 | 0.7353  |  0.8713 |
| Sinhala-English (NMT)   |  Direct    | TransQuest          | [model.zip](https://bit.ly/3dKM3ki)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6525 | 0.4510  |  0.5570 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/3foBSlP)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.5957 | 0.5078  |  0.6466 |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.3737 | 0.7517  |  0.8978 |
| Russian-English (NMT)   |  Direct    | TransQuest          | [model.zip](https://bit.ly/30lMA8c)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.7734 | 0.5076  |  0.6856 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/2B3UM2D)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.5479 | 0.8253  |  1.1930 |
| English-German (NMT)    |  Direct    | TransQuest          | [model.zip](https://bit.ly/2UpFiwF)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.4669 | 0.6474  |  0.7762 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/3d8gT5n)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.1455 | 0.6791  |  0.9670 |
|                         |  HTER      | TransQuest          | [model.zip](https://bit.ly/37tkTvZ)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.4994 | 0.1486  |  0.1842 |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/3icI5Dw)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.3916 | 0.1500  |  0.1896 |
| English-Chinese (NMT)   |  Direct    | TransQuest          | [model.zip](https://bit.ly/2XGAx3Q)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.4779 | 0.9865  | 1.1338  |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/3h4WSQ8)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.4067 | 1.0389  | 1.1973  |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.1676 | 0.6559  | 0.8503  |
|                         |  HTER      | TransQuest          | [model.zip](https://bit.ly/3ge3wSN)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.5910 | 0.1400  | 0.1717  |
|                         |            | SiameseTransQuest   | [model.zip](https://bit.ly/2YLIvJw)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                         |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.5058 | 0.1470  | 0.1814  |
| English-Latvian (SMT)   |  HTER      | TransQuest          | [model.zip](https://bit.ly/3gkY7JX)  | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |  0.7141 | 0.1041  |  0.1420 |
|                         |            | SiameseTransQuest   |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |         |         |         |
|                         |            | Quest++             |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |  0.3528 | 0.1554  |  0.1919 |
| English-Latvian (NMT)   |  HTER      | TransQuest          | [model.zip](https://bit.ly/3eLb1jU)  | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |  0.7450 | 0.1162  |  0.1601 |
|                         |            | SiameseTransQuest   |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |         |         |         |
|                         |            | Quest++             |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.4435 | 0.1625  |  0.2164 |
| English-German (SMT)    |  HTER      | TransQuest          | [model.zip](https://bit.ly/3dNafBx)  | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.7355 | 0.0967  |  0.1300 |
|                         |            | SiameseTransQuest   |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |         |         |         |
|                         |            | Quest++             |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.3653 | 0.1402  |  0.1772 |
| English-Czech (SMT)     |  HTER      | TransQuest          | [model.zip](https://bit.ly/2VyBOZ2)  | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.7150 | 0.1198  |  0.1611 |
|                         |            | SiameseTransQuest   |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |         |         |         |
|                         |            | Quest++             |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.3943 | 0.1651  |  0.2110 |
| German-English (SMT)    |  HTER      | TransQuest          | [model.zip](https://bit.ly/3dRlqJu)  | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.7878 | 0.0934  |  0.1277 |
|                         |            | SiameseTransQuest   |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) |         |         |         |
|                         |            | Quest++             |                                      | [WMT 2018](https://www.statmt.org/wmt18/quality-estimation-task.html) | 0.3323 | 0.1508  |  0.1928 |
    
Once downloading them and unzipping it, they can be loaded easily

```bash
model = QuestModel("xlmroberta", "path, num_labels=1,
                               use_cuda=torch.cuda.is_available(), args=transformer_config)
```

```bash
model = SiameseTransQuestModel("path")
``` 

## Citation
Please consider citing us if you use the library. 
```bash
Coming soon!
Please keep in touch
```