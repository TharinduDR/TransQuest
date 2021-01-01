## [WMT 2020 QE Task 2: Sentence-Level Post-editing Effort](http://www.statmt.org/wmt20/quality-estimation-task.html)
This task consists predicting Sentence-level HTER (Human Translation Error Rate) scores for a given source and a target. 

To run the experiments for each language please run this command from the root directory of TransQuest.  

```bash
python -m examples.sentence_level.wmt_2020_task2.<language-pair>.<architecture>
```

Language Pair options :  en_zh (English-Chinese), en_de (English-German)

Architecture Options : monotransquest (MonoTransQuest), siamesetransquest (SiameseTransQuest).

As an example to run the experiments on English-Chinese with MonoTransQuest architecture, run the following command. 

```bash
python -m examples.sentence_level.wmt_2020_task2.en_zh.trans_quest
```

### Results
Both architectures in TransQuest outperforms OpenKiwi in all the language pairs. 

| Language Pair           |     Algorithm        |  Pearson | MAE      | RMSE      |
|:-----------------------:|--------------------- | -------: | --------:| --------: |  
| English-German          |**MonoTransQuest**    |**0.4994**|**0.1486**|**0.1842** |
|                         |  SiameseTransQuest   |  0.4875  | 0.1501   |  0.1886   |
|                         |  OpenKiwi            |  0.3916  | 0.1500   |  0.1896   |
| English-Chinese         |**MonoTransQuest**    |**0.5910**|**0.1351**|**0.1681** |
|                         |  SiameseTransQuest   |  0.5621  | 0.1411   | 0.1723    |
|                         |  OpenKiwi            |  0.5058  | 0.1470   | 0.1814    |
