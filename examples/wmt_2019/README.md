## [WMT 2019 QE Task 1: Sentence-Level QE](http://www.statmt.org/wmt19/qe-task.html)
The participating systems are expected to predict the sentence-level HTER score (the percentage of edits needed to fix the translation)

To run the experiments for each language, please run this command from the root directory of TransQuest.  

```bash
python -m examples.wmt_2019.<language-pair>.<architecture>
```

Language Pair options :  en_ru (English-Russian), en_de (English-German)

Architecture Options : trans_quest (MonoTransQuest), siamese_trans_quest (SiameseTransQuest).

As an example to run the experiments on English-Russian with MonoTransQuest architecture, run the following command. 

```bash
python -m examples.wmt_2019.en_ru.trans_quest
```

### Results
Both architectures in TransQuest outperforms QuEst++  in all the language pairs.

| Language Pair           |     Algorithm        |  Pearson | 
|:-----------------------:|--------------------- | -------: | 
| English-German          |**MonoTransQuest**    |**0.5117**|
|                         |  SiameseTransQuest   |  0.4951  |
|                         |  QuEst++             |  0.4001  | 
| English-Russian         |**MonoTransQuest**    |**0.7126**|
|                         |  SiameseTransQuest   |  0.6432  |
|                         |  QuEst++             |  0.2601  |