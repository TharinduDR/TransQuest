## [WMT 2018 QE Task 1: Sentence-Level QE](https://www.statmt.org/wmt18/quality-estimation-task.html)
The participating systems are expected to predict the sentence-level HTER score (the percentage of edits needed to fix the translation)

To run the experiments for each language, please run this command from the root directory of TransQuest. If both NMT and SMT is available for a certain language pair, specify that too.  

```bash
python -m examples.wmt_2019.<language-pair>.<nmt/smt><architecture>
```

Language Pair options :  en_de (English-German) (both NMT and SMT), en_lv(English-Latvian) (both NMT and SMT), en_cs(English-Czech), de_en 

Architecture Options : monotransquest (MonoTransQuest), siamesetransquest (SiameseTransQuest).

As an example to run the experiments on English-Latvian NMT with MonoTransQuest architecture, run the following command. 

```bash
python -m examples.wmt_2018.en_lv.nmt.trans_quest
```

To run the English-Czech experiments with MonoTransQuest architecture,, run the following command

```bash
python -m examples.wmt_2018.en_cs.trans_quest
```


### Results
Both architectures in TransQuest outperforms QuEst++ in all the language pairs. 

| Language Pair           |     Algorithm        |  Pearson | MAE      | RMSE      |
|:-----------------------:|--------------------- | -------: | --------:| --------: |  
| English-German (NMT)    |**MonoTransQuest**    |**0.4784**|**0.1264**|**0.1770** |
|                         |  SiameseTransQuest   |  0.4152  | 0.1270   |  0.1796   |
|                         |  QuEst++             |  0.2874  | 0.1286   |  0.1886   |
| English-German (SMT)    |**MonoTransQuest**    |**0.7355**|**0.0967**|**0.1300** |
|                         |  SiameseTransQuest   |  0.6992  | 0.1258   |  0.1438   |
|                         |  QuEst++             |  0.3653  | 0.1402   |  0.1772   |
| English-Latvian (NMT)   |**MonoTransQuest**    |**0.7450**|**0.1162**|**0.1601** |
|                         |  SiameseTransQuest   |  0.7183  | 0.1456   |  0.1892   |
|                         |  QuEst++             |  0.4435  | 0.1625   |  0.2164   |
| English-Latvian (SMT)   |**MonoTransQuest**    |**0.7141**|**0.1041**|**0.1420** |
|                         |  SiameseTransQuest   |  0.6320  | 0.1274   |  0.1661   |
|                         |  QuEst++             |  0.3528  | 0.1554   |  0.1919   |
| English-Czech           |**MonoTransQuest**    |**0.7207**|**0.1197**|**0.1631** |
|                         |  SiameseTransQuest   |  0.6853  | 0.1298   |  0.1801   |
|                         |  QuEst++             |  0.3943  | 0.1651   |  0.2110   |
| German-English          |**MonoTransQuest**    |**0.7939**|**0.0934**|**0.1277** |
|                         |  SiameseTransQuest   |  0.7524  | 0.1194   |  0.1502   |
|                         |  QuEst++             |  0.3323  | 0.1508   |  0.1928   |