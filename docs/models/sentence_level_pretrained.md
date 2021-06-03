# Sentence Level Pre-trained Models
We have released several pre-trained TransQuest models on two aspects in sentence-level quality estimation. We will be keep releasing new models. So please keep in touch.

## Predicting Direct Assessment
The current practice in MT evaluation is the so-called Direct Assessment (DA) of MT quality, where raters evaluate the machine translation on a continuous 1-100 scale. This method has been shown to improve the reproducibility of manual evaluation and to provide a more reliable gold standard for automatic evaluation metrics

We have released several quality estimation models for this aspect. We have also released a couple of multi-language pair models that would work on any language pair in any domain. 

### Available Models

| Language Pair   | NMT/SMT        |  Domain      |     Algorithm       |  Model Link                          | 
|:---------------:|:--------------:|:------------:|:-------------------:|:------------------------------------:|
| Romanian-English| NMT            |  Wikipedia   | MonoTransQuest      | [TransQuest/monotransquest-da-ro_en-wiki](https://huggingface.co/TransQuest/monotransquest-da-ro_en-wiki)  |
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-ro_en-wiki](https://huggingface.co/TransQuest/siamesetransquest-da-ro_en-wiki)  |
| Estonian-English| NMT            | Wikipedia    | MonoTransQuest      | [TransQuest/monotransquest-da-et_en-wiki](https://huggingface.co/TransQuest/monotransquest-da-et_en-wiki)  | 
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-et_en-wiki](https://huggingface.co/TransQuest/siamesetransquest-da-et_en-wiki)  | 
| Nepalese-English| NMT            | Wikipedia    | MonoTransQuest      | [TransQuest/monotransquest-da-ne_en-wiki](https://huggingface.co/TransQuest/monotransquest-da-ne_en-wiki)  | 
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-ne_en-wiki](https://huggingface.co/TransQuest/siamesetransquest-da-ne_en-wiki)  | 
| Sinhala-English | NMT            |  Wikipedia   | MonoTransQuest      | [TransQuest/monotransquest-da-ne_en-wiki](https://huggingface.co/TransQuest/monotransquest-da-ne_en-wiki)  | 
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-si_en-wiki](https://huggingface.co/TransQuest/siamesetransquest-da-si_en-wiki)  | 
| Russian-English | NMT            | Wikipedia    | MonoTransQuest      | [TransQuest/monotransquest-da-ru_en-reddit_wikiquotes](https://huggingface.co/TransQuest/monotransquest-da-ru_en-reddit_wikiquotes)  | 
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-ru_en-reddit_wikiquotes](https://huggingface.co/TransQuest/siamesetransquest-da-ru_en-reddit_wikiquotes)  | 
| English-German  | NMT            | Wikipedia    | MonoTransQuest      | [TransQuest/monotransquest-da-en_de-wiki](https://huggingface.co/TransQuest/monotransquest-da-en_de-wiki)  | 
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-en_de-wiki](https://huggingface.co/TransQuest/siamesetransquest-da-en_de-wiki)  | 
| English-Chinese | NMT            | Wikipedia    | MonoTransQuest      | [TransQuest/monotransquest-da-en_zh-wiki](https://huggingface.co/TransQuest/monotransquest-da-en_zh-wiki)  | 
|                 |                |              | SiameseTransQuest   | [TransQuest/siamesetransquest-da-en_zh-wiki](https://huggingface.co/TransQuest/siamesetransquest-da-en_zh-wiki)  | 
| English-\*      | Any            | Any          | MonoTransQuest      | [TransQuest/monotransquest-da-en_any](https://huggingface.co/TransQuest/monotransquest-da-en_any)   | 
|                 |                |              | SiameseTransQuest   |                                      | 
| \*-English      | Any            | Any          | MonoTransQuest      | [TransQuest/monotransquest-da-any_en](https://huggingface.co/TransQuest/monotransquest-da-any_en)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| \*-\*           | Any            | Any          | MonoTransQuest      | [TransQuest/monotransquest-da-multilingual](https://huggingface.co/TransQuest/monotransquest-da-multilingual)                                     | 
|                 |                |              | SiameseTransQuest   |                                     | 

!!! note
    \* denotes any language. (\*-\* means any language to any language)

## Predicting HTER
The performance of QE systems has typically been assessed using the semiautomatic HTER (Human-mediated Translation Edit Rate). HTER is an edit-distance-based measure which captures the distance between the automatic translation and a reference translation in terms of the number of modifications required to transform one into another. In light of this, a QE system should be able to predict the percentage of edits required in the translation. 

We have released several quality estimation models for this aspect. We have also released a couple of multi-language pair models that would work on any language pair in any domain. 

### Available Models

| Language Pair   | NMT/SMT        |  Domain      |     Algorithm       |  Model Link                          | 
|:---------------:|:--------------:|:------------:|:-------------------:|:------------------------------------:|
| English-German  | NMT            |  Wikipedia   | MonoTransQuest      | [TransQuest/monotransquest-hter-en_de-wiki](https://huggingface.co/TransQuest/monotransquest-hter-en_de-wiki)  |
|                 |                |              | SiameseTransQuest   |                                      |
|                 | NMT            |   IT         | MonoTransQuest      | [TransQuest/monotransquest-hter-en_de-it-nmt](https://huggingface.co/TransQuest/monotransquest-hter-en_de-it-nmt)  | 
|                 |                |              | SiameseTransQuest   |                                      |  
|                 | SMT            |   IT         | MonoTransQuest      | [TransQuest/monotransquest-hter-en_de-it-smt](https://huggingface.co/TransQuest/monotransquest-hter-en_de-it-smt)  | 
|                 |                |              | SiameseTransQuest   |                                      |  
| English-Latvian | SMT            | Life Sciences| MonoTransQuest      | [TransQuest/monotransquest-hter-en_lv-it-nmt](https://huggingface.co/TransQuest/monotransquest-hter-en_lv-it-nmt)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
|                 | NMT            | Life Sciences| MonoTransQuest      | [TransQuest/monotransquest-hter-en_lv-it-smt](https://huggingface.co/TransQuest/monotransquest-hter-en_lv-it-smt)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-Czech   | SMT            |  IT          | MonoTransQuest      | [TransQuest/monotransquest-hter-en_cs-pharmaceutical](https://huggingface.co/TransQuest/monotransquest-hter-en_cs-pharmaceutical)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| German-English  | SMT            | Life Sciences| MonoTransQuest      | [TransQuest/monotransquest-hter-de_en-pharmaceutical](https://huggingface.co/TransQuest/monotransquest-hter-de_en-pharmaceutical)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-Chinese | NMT            | Wikipedia    | MonoTransQuest      | [TransQuest/monotransquest-hter-en_zh-wiki](https://huggingface.co/TransQuest/monotransquest-hter-en_zh-wiki)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-*       | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 
| \*-\*           | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 

!!! note
    \* denotes any language. (\*-\* means any language to any language)