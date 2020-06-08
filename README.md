[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/transquest)](https://pepy.tech/project/transquest)

# TransQuest : Transformer based Translation Quality Estimation. 

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

## Minimal Start
Once TransQuest is properly installed you can train your own models using the following scripts. For more information, please refer the "Run the examples" section.

### TransQuest Model
```bash
model = QuestModel("xlmroberta", "xlm-roberta-large", num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=transformer_config)
model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
predictions, raw_outputs = model.predict(test_sentence_pairs)
```

### SiameseTransQuest Model
```bash
pip install transquest
```

## Run the examples
Examples are included in the repository but are not shipped with the library.

1. [WMT 2020 Sentence-level Direct Assessment QE Shared Task](examples/wmt_2020)
2. [WMT 2020 Sentence-level Post-Editing Effort QE Shared Task](examples/wmt_2020_task2)
3. [WMT 2019 Sentence-level Post-Editing Effort QE Shared Task](examples/wmt_2019)

## Pre-trained models
Following pre-trained models are released. We will be keep releasing new models. Please keep in touch. 

| Language Pair    |  Objective |     Algorithm       |  Model Link                          | Data                                                                 | Pearson | MAE     | RMSE    |
|:----------------:|----------- |:-------------------:|:------------------------------------:|:--------------------------------------------------------------------:| ------: | ------: | ------: |  
| Romanian-English |  Direct    | TransQuest          | [model.zip](https://bit.ly/2AfuXwb)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.8982 | 0.3121  |  0.4097 |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.8501 | 0.3637  |  0.4932 |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6845 | 0.7596  |  1.0522 |
| Estonian-English |  Direct    | TransQuest          | [model.zip](https://bit.ly/2YjXIAa)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.7748 | 0.5904  |  0.7321 |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6804 | 0.7047  |  0.9022 |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.4770 | 0.9176  |  1.1382 |
| Nepalese-English |  Direct    | TransQuest          | [model.zip](https://bit.ly/2MHnCZc)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.7914 | 0.3975  |  0.5078 |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.3860 | 0.7353  |  0.8713 |
| Sinhala-English  |  Direct    | TransQuest          | [model.zip](https://bit.ly/3dKM3ki)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.6525 | 0.4510  |  0.5570 |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |  0.3737 | 0.7517  |  0.8978 |
| Russian-English  |  Direct    | TransQuest          | [model.zip](https://bit.ly/30lMA8c)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
| English-German   |  Direct    | TransQuest          | [model.zip](https://bit.ly/2UpFiwF)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |  HTER      | TransQuest          |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |  HTER      | TransQuest          |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
| English-Chinese  |  Direct    | TransQuest          | [model.zip](https://bit.ly/2XGAx3Q)  | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |  HTER      | TransQuest          |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | SiameseTransQuest   |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
|                  |            | OpenKiwi            |                                      | [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html) |         |         |         |
  
 

## Citation
Please consider citing us if you use the library. 
```bash
Coming soon!
Please keep in touch
```