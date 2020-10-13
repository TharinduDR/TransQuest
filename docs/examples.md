# Examples
We have provided several examples on how to use TransQuest in recent WMT sentence-level quality estimation shared tasks. They are included in the repository but are not shipped with the library. Therefore, if you need to run the examples, please clone the repository. 

```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```

In the examples folder you will find the following tasks.

## [WMT 2020 QE Task 1: Sentence-Level Direct Assessment](http://www.statmt.org/wmt20/quality-estimation-task.html)
The participants were predict the direct assessment of a source and a target. There were seven language-pairs released by the organisers. 

To run the experiments for each language please run this command from the root directory of TransQuest.  

```bash
python -m examples.wmt_2020.<language-pair>.<architecture>
```

Language Pair options :  ro_en (Romanian-English), ru_en (Russian-English), et_en (Estonian-English), en_zh (English-Chinese), ne_en (Nepalese-English), en_de (English-German), si_en(Sinhala-English)

Architecture Options : trans_quest (MonoTransQuest), siamese_trans_quest (SiameseTransQuest).

As an example to run the experiments on Romanian-English with MonoTransQuest architecture, run the following command. 

```bash
python -m examples.wmt_2020.ro_en.trans_quest
```

### Results
Both architectures in TransQuest outperforms OpenKiwi in all the language pairs. Furthermore, TransQuest won this task in all the language pairs. 

| Language Pair           |     Algorithm        |  Pearson | MAE      | RMSE      |
|:-----------------------:|--------------------- | -------: | --------:| --------: |  
| Romanian-English        |  **MonoTransQuest**  |**0.8982**|**0.3121**|**0.4097** |
|                         |  SiameseTransQuest   |  0.8501  | 0.3637   |  0.4932   |
|                         |  OpenKiwi            |  0.6845  | 0.7596   |  1.0522   |
| Estonian-English        |**MonoTransQuest**    |**0.7748**|**0.5904**|**0.7321** |
|                         |  SiameseTransQuest   |  0.6804  | 0.7047   |  0.9022   |
|                         |  OpenKiwi            |  0.4770  | 0.9176   |  1.1382   |
| Nepalese-English        |**MonoTransQuest**    |**0.7914**|**0.3975**|**0.5078** |
|                         |  SiameseTransQuest   |  0.6081  | 0.6531   |  0.7950   |
|                         |  OpenKiwi            |  0.3860  | 0.7353   |  0.8713   |
| Sinhala-English         |**MonoTransQuest**    |**0.6525**|**0.4510**|**0.5570** |
|                         |  SiameseTransQuest   |  0.5957  | 0.5078   |  0.6466   |
|                         |  OpenKiwi            |  0.3737  | 0.7517   |  0.8978   |
| Russian-English         |**MonoTransQuest**    |**0.7734**|**0.5076**|**0.6856** |
|                         |  SiameseTransQuest   |          |          |           |
|                         |  OpenKiwi            |  0.5479  | 0.8253   |  1.1930   |
| English-German          |**MonoTransQuest**    |**0.4669**|**0.6474**|  0.7762   |
|                         |  SiameseTransQuest   |          |          |           |
|                         |  OpenKiwi            |  0.1455  | 0.6791   |  0.9670   |
| English-Chinese         |**MonoTransQuest**    |**0.4779**|**0.9865**|**1.1338** |
|                         |  SiameseTransQuest   |  0.4067  | 1.0389   | 1.1973    |
|                         |  OpenKiwi            |  0.1676  | 0.6559   | 0.8503    |


## [WMT 2020 QE Task 2: Sentence-Level Post-editing Effort](http://www.statmt.org/wmt20/quality-estimation-task.html)
This task consists predicting Sentence-level HTER (Human Translation Error Rate) scores for a given source and a target. 

To run the experiments for each language please run this command from the root directory of TransQuest.  

```bash
python -m examples.wmt_2020_task2.<language-pair>.<architecture>
```

Language Pair options :  en_zh (English-Chinese), en_de (English-German)

Architecture Options : trans_quest (MonoTransQuest), siamese_trans_quest (SiameseTransQuest).

As an example to run the experiments on English-Chinese with MonoTransQuest architecture, run the following command. 

```bash
python -m examples.wmt_2020_task2.en_zh.trans_quest
```

### Results
Both architectures in TransQuest outperforms OpenKiwi in all the language pairs. Furthermore, TransQuest won this task in all the language pairs. 

| Language Pair           |     Algorithm        |  Pearson | MAE      | RMSE      |
|:-----------------------:|--------------------- | -------: | --------:| --------: |  
| English-German          |**MonoTransQuest**    |**0.4994**|**0.1486**|  0.1842   |
|                         |  SiameseTransQuest   |          |          |           |
|                         |  OpenKiwi            |  0.3916  | 0.1500   |  0.1896   |
| English-Chinese         |**MonoTransQuest**    |**0.6119**|**0.1351**|**0.1681** |
|                         |  SiameseTransQuest   |          |          |           |
|                         |  OpenKiwi            |  0.5058  | 0.1470   | 0.1814    |

## [WMT 2019 QE Task 1: Sentence-Level QE](http://www.statmt.org/wmt20/quality-estimation-task.html)
TThe participating systems are expected to predict the sentence-level HTER score (the percentage of edits needed to fix the translation)

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
Both architectures in TransQuest outperforms OpenKiwi in all the language pairs. Furthermore, TransQuest won this task in all the language pairs. 

| Language Pair           |     Algorithm        |  Pearson | MAE      | RMSE      |
|:-----------------------:|--------------------- | -------: | --------:| --------: |  
| English-German          |**MonoTransQuest**    |**0.4994**|**0.1486**|  0.1842   |
|                         |  SiameseTransQuest   |          |          |           |
|                         |  OpenKiwi            |  0.3916  | 0.1500   |  0.1896   |
| English-Russian         |**MonoTransQuest**    |**0.6119**|**0.1351**|**0.1681** |
|                         |  SiameseTransQuest   |          |          |           |
|                         |  OpenKiwi            |  0.5058  | 0.1470   | 0.1814    |

