# TransQuest : Transformer based Translation Quality Estimation. 

TransQuest provides state-of-the-art models for Quality Estimation.

## Installation
you first need to install PyTorch.
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
python -m examples.ro_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM               | xlm-mlm-enro-1024            | 0.739               | 
TransQuest  | BERT              | bert-base-multilingual-cased | 0.829               | 


### Russian - English 
```bash
python -m examples.ru_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.692               | 
TransQuest  | BERT              | bert-base-multilingual-cased | 0.642               | 

### Estonian - English 
```bash
python -m examples.et_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.672               | 
TransQuest  | BERT              | bert-base-multilingual-cased | 0.664               | 

### English - Chinese
```bash
python -m examples.en_zh.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.493               |
TransQuest  | BERT              | bert-base-multilingual-cased | 0.518               |  


### Nepalese - English 
```bash
python -m examples.si_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.699               | 
TransQuest  | BERT              | bert-base-multilingual-cased | 0.684               | 


### English - German 
```bash
python -m examples.en_de.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name             | Pearson Correlation | 
------------| ----------------- |-----------------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base             | 0.326               | 
TransQuest  | BERT              | bert-base-multilingual-cased | 0.449               | 


### Sinhala - English 
```bash
python -m examples.si_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name  | Pearson Correlation | 
------------| ----------------- |------------------:| -------------------:| 
TransQuest  | XLM-R             | xlm-roberta-base  | 0.380               | 