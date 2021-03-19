# Word Level Examples
We have provided several examples on how to use TransQuest in recent WMT word-level quality estimation shared tasks. They are included in the repository but are not shipped with the library. Therefore, if you need to run the examples, please clone the repository.

!!! note
Please don't use the same environment you used to install TransQuest to run the examples. Create a new environment. 

```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```

In the examples/word_level folder you will find the following tasks.

## [WMT 2020 QE Task 2: Word-Level Post-editing Effort](http://www.statmt.org/wmt20/quality-estimation-task.html)
This task consists predicting Word-level quality for a given source and a target. It requires predicting word level quality in source and target as OK, BAD and also the quality of the "gaps" in target as Ok, BAD.

To run the experiments for each language please run this command from the root directory of TransQuest.  

```bash
python -m examples.word_level.wmt_2020.<language-pair>.<architecture>
```

Language Pair options :  en_zh (English-Chinese), en_de (English-German)

Architecture Options : microtransquest (MicroTransQuest)

As an example to run the experiments on English-Chinese with MicroTransQuwst architecture, run the following command. 

```bash
python -m examples.word_level.wmt_2020.en_zh.microtransquest
```

### Results
MicroTransQuest architecture in TransQuest outperforms OpenKiwi in all the language pairs. 

| Language Pair           |     Algorithm        |  Source <br /> F1 Multi | Target <br /> F1 Multi      | 
|:-----------------------:|--------------------- | ----------------------: | ---------------------------:| 
| English-German          |**MicroTransQuest**   |**0.5456**               |**0.6013**|
|                         |  OpenKiwi            |  0.3717                 | 0.4111   |
| English-Chinese         |**MicroTransQuest**   |**0.4440**               |**0.6402**|
|                         |  OpenKiwi            |  0.3729                 | 0.5583   | 

## [WMT 2019 QE Task 2: Word-Level QE](http://www.statmt.org/wmt19/qe-task.html)
The participating systems are expected to predict the  Word-level quality for a given source and a target.

To run the experiments for each language, please run this command from the root directory of TransQuest.  

```bash
python -m examples.sentence_level.wmt_2019.<language-pair>.<architecture>
```

Language Pair options :  en_ru (English-Russian)

Architecture Options : microtransquest (MicroTransQuest)

As an example to run the experiments on English-Russian with MicroTransQuest architecture, run the following command. 

```bash
python -m examples.word_level.wmt_2019.en_ru.microtransquest
```

### Results
MicroTransQuest architecture in TransQuest outperforms OpenKiwi  in En-Ru.

| Language Pair           |     Algorithm        |  Source <br /> F1 Multi | Target <br /> F1 Multi  | 
|:-----------------------:|--------------------- | ----------------------: | ----------------------: | 
| English-Russian         |**MicroTransQuest**   |**0.5543**               |**0.5592**               |
|                         |  OpenKiwi            |  0.2647                 |  0.2412                 |

## [WMT 2018 QE Task 2: Word-Level QE](https://www.statmt.org/wmt18/quality-estimation-task.html)
The participating systems are expected to predict the  Word-level quality for a given source and a target.

To run the experiments for each language, please run this command from the root directory of TransQuest. If both NMT and SMT is available for a certain language pair, specify that too.  

```bash
python -m examples.word_level.wmt_2019.<language-pair>.<nmt/smt><architecture>
```

Language Pair options :  en_de (English-German) (both NMT and SMT), en_lv(English-Latvian) (both NMT and SMT), en_cs(English-Czech), de_en 

Architecture Options : microtransquest (MicroTransQuest)

As an example to run the experiments on English-Latvian NMT with MicroTransQuest architecture, run the following command. 

```bash
python -m examples.word_level.wmt_2018.en_lv.nmt.microtransquest
```

To run the English-Czech experiments with MicroTransQuest architecture,, run the following command

```bash
python -m examples.word_level.wmt_2018.en_cs.microtransquest
```


### Results
MicroTransQuest architecture in TransQuest outperforms Marmot in all the language pairs. 

| Language Pair           |     Algorithm        |   Source <br /> F1 Multi | Target <br /> F1 Multi  | Gaps <br /> F1 Multi      |
|:-----------------------:|--------------------- | -----------------------: | -----------------------:| ------------------------: |  
| English-German (NMT)    |**MicroTransQuest**   |**0.2957**                |**0.4421**               |**0.1672** |
|                         |  Marmot              |  0.0000                  | 0.1812                  |  0.0000   |
| English-German (SMT)    |**MicroTransQuest**   |**0.5269**                |**0.6348**               |**0.4927** |
|                         |  Marmot              |  0.0000                  | 0.3630                  |  0.0000   |
| English-Latvian (NMT)   |**MicroTransQuest**   |**0.4880**                |**0.5868**               |**0.1664** |
|                         |  Marmot              |  0.0000                  | 0.4208                  |  0.0000   |
| English-Latvian (SMT)   |**MicroTransQuest**   |**0.4945**                |**0.5939**               |**0.2356** |
|                         |  Marmot              |  0.0000                  | 0.3445                  |  0.0000   |
| English-Czech           |**MicroTransQuest**   |**0.5327**                |**0.6081**               |**0.2018** |
|                         |  Marmot              |  0.0000                  | 0.4449                  |  0.0000   |
| German-English          |**MicroTransQuest**   |**0.4824**                |**0.6485**               |**0.4203** |
|                         |  Marmot              |  0.0000                  | 0.4373                  |  0.0000   |

!!! note
Please note that in WMT 2018 the organisers evaluated the gaps and the words in MT separately. This is different from WMT 2019 and WMT 2020.

!!! note
Please note that the baseline used in WMT 2018; Marmot does not support predicting quality for words in source and gaps in target. Hence, those values are set to 0.0000 in all the language pairs.
