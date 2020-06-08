## [WMT 2020 Sentence-level Direct Assessment QE Shared Task](https://competitions.codalab.org/competitions/24447) 
Following sections show the results for all the experimented configurations in TransQuest and SiameseTransQuest


### Romanian - English 
```bash
python -m examples.wmt_2020.ro_en.trans_quest
```

Algo Type   | Transformer Type  | Transformer Name                   | Pearson Correlation | 
------------| ----------------- |-----------------------------------:| -------------------:| 
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