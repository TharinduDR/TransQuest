[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/transquest)](https://pepy.tech/project/transquest)

# TransQuest : Transformer based Translation Quality Estimation. 

TransQuest provides state-of-the-art models for Quality Estimation.

## Installation
you first need to install PyTorch. THe recommended PyTorch version is 1.5.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```

## Run the examples
Examples are included in the repository but are not shipped with the library.

### Romanian - English 
```bash
python -m examples.wmt_2020.ro_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name                   | Pearson Correlation | 
------------| ----------------- |-----------------------------------:| -------------------:| 
TransQuest  | XLM               | xlm-mlm-enro-1024                  | 0.739               |
TransQuest  | XLM-R             | xlm-roberta-large                  | 0.894               |  
TransQuest  | BERT              | bert-base-multilingual-cased       | 0.829               | 
TransQuest  | DistilBERT        | distilbert-base-multilingual-cased | 0.778               | 


### Russian - English 
```bash
python -m examples.wmt_2020.ru_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name                   | Pearson Correlation | 
------------| ----------------- |-----------------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base                   | 0.692               | 
TransQuest  | XLM-R             | xlm-roberta-large                  | 0.715               | 
TransQuest  | BERT              | bert-base-multilingual-cased       | 0.642               |
TransQuest  | DistilBERT        | distilbert-base-multilingual-cased | 0.644               | 

### Estonian - English 
```bash
python -m examples.wmt_2020.et_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.672               |
TransQuest  | XLM-R             | xlm-roberta-large            | 0.741               |  
TransQuest  | BERT              | bert-base-multilingual-cased | 0.664               | 

### English - Chinese
```bash
python -m examples.wmt_2020.en_zh.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.493               |
TransQuest  | XLM-R             | xlm-roberta-large            | 0.526               |
TransQuest  | BERT              | bert-base-multilingual-cased | 0.518               |  


### Nepalese - English 
```bash
python -m examples.wmt_2020.si_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.699               |
TransQuest  | XLM-R             | xlm-roberta-large            | 0.761               |  
TransQuest  | BERT              | bert-base-multilingual-cased | 0.684               | 


### English - German 
```bash
python -m examples.wmt_2020.en_de.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM               | xlm-mlm-ende-1024            | 0.326               | 
TransQuest  | XLM-R             | xlm-roberta-large            | 0.475               |
TransQuest  | BERT              | bert-base-multilingual-cased | 0.449               | 


### Sinhala - English 
```bash
python -m examples.wmt_2020.si_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name   | Pearson Correlation | 
------------| ----------------- |-------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base   | 0.380               | 
TransQuest  | XLM-R             | xlm-roberta-large  | 0.589               | 