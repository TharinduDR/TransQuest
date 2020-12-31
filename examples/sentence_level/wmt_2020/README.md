## [WMT 2020 QE Task 1: Sentence-Level Direct Assessment](http://www.statmt.org/wmt20/quality-estimation-task.html)
The participants were predict the direct assessment of a source and a target. There were seven language-pairs released by the organisers. 

To run the experiments for each language please run this command from the root directory of TransQuest.  

```bash
python -m examples.sentenceee_level.wmt_2020.<language-pair>.<architecture>
```

Language Pair options :  ro_en (Romanian-English), ru_en (Russian-English), et_en (Estonian-English), en_zh (English-Chinese), ne_en (Nepalese-English), en_de (English-German), si_en(Sinhala-English)

Architecture Options : monotransquest (MonoTransQuest), siamesetransquest (SiameseTransQuest).

As an example to run the experiments on Romanian-English with MonoTransQuest architecture, run the following command. 

```bash
python -m examples.wmt_2020.ro_en.monotransquest
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
|                         |  SiameseTransQuest   |  0.7126  | 0.6132   |  0.8531   |
|                         |  OpenKiwi            |  0.5479  | 0.8253   |  1.1930   |
| English-German          |**MonoTransQuest**    |**0.4669**|**0.6474**|**0.7762** |
|                         |  SiameseTransQuest   |  0.3992  |  0.6651  |  0.8497   |
|                         |  OpenKiwi            |  0.1455  | 0.6791   |  0.9670   |
| English-Chinese         |**MonoTransQuest**    |**0.4779**|**0.9865**|**1.1338** |
|                         |  SiameseTransQuest   |  0.4067  | 1.0389   | 1.1973    |
|                         |  OpenKiwi            |  0.1676  | 0.6559   | 0.8503    |