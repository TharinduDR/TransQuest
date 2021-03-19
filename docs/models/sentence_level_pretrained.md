# Sentence Level Pre-trained Models
We have released several pre-trained TransQuest models on two aspects in sentence-level quality estimation. We will be keep releasing new models. So please keep in touch.

## Predicting Direct Assessment
The current practice in MT evaluation is the so-called Direct Assessment (DA) of MT quality, where raters evaluate the machine translation on a continuous 1-100 scale. This method has been shown to improve the reproducibility of manual evaluation and to provide a more reliable gold standard for automatic evaluation metrics

We have released several quality estimation models for this aspect. We have also released a couple of multi-language pair models that would work on any language pair in any domain. 

### Available Models

| Language Pair   | NMT/SMT        |  Domain      |     Algorithm       |  Model Link                          | 
|:---------------:|:--------------:|:------------:|:-------------------:|:------------------------------------:|
| Romanian-English| NMT            |  Wikipedia   | MonoTransQuest      | [model.zip](https://bit.ly/2AfuXwb)  |
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/37vT4mt)  |
| Estonian-English| NMT            | Wikipedia    | MonoTransQuest      | [model.zip](https://bit.ly/2YjXIAa)  | 
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/30mO5mW)  | 
| Nepalese-English| NMT            | Wikipedia    | MonoTransQuest      | [model.zip](https://bit.ly/2MHnCZc)  | 
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/3h674bc)  | 
| Sinhala-English | NMT            |  Wikipedia   | MonoTransQuest      | [model.zip](https://bit.ly/3dKM3ki)  | 
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/3foBSlP)  | 
| Russian-English | NMT            | Wikipedia    | MonoTransQuest      | [model.zip](https://bit.ly/30lMA8c)  | 
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/2B3UM2D)  | 
| English-German  | NMT            | Wikipedia    | MonoTransQuest      | [model.zip](https://bit.ly/2UpFiwF)  | 
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/3d8gT5n)  | 
| English-Chinese | NMT            | Wikipedia    | MonoTransQuest      | [model.zip](https://bit.ly/2XGAx3Q)  | 
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/3h4WSQ8)  | 
| English-\*      | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 
| \*-English      | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 
| \*-\*           | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 

!!! note
\* denotes any language. (\*-\* means any language to any language)

## Predicting HTER
The performance of QE systems has typically been assessed using the semiautomatic HTER (Human-mediated Translation Edit Rate). HTER is an edit-distance-based measure which captures the distance between the automatic translation and a reference translation in terms of the number of modifications required to transform one into another. In light of this, a QE system should be able to predict the percentage of edits required in the translation. 

We have released several quality estimation models for this aspect. We have also released a couple of multi-language pair models that would work on any language pair in any domain. 

### Available Models

| Language Pair   | NMT/SMT        |  Domain      |     Algorithm       |  Model Link                          | 
|:---------------:|:--------------:|:------------:|:-------------------:|:------------------------------------:|
| English-German  | NMT            |  Wikipedia   | MonoTransQuest      | [model.zip](https://bit.ly/37tkTvZ)  |
|                 |                |              | SiameseTransQuest   | [model.zip](https://bit.ly/3icI5Dw)  |
|                 | SMT            |   IT         | MonoTransQuest      | [model.zip](https://bit.ly/3dNafBx)  | 
|                 |                |              | SiameseTransQuest   |                                      |  
| English-Latvian | SMT            | Life Sciences| MonoTransQuest      | [model.zip](https://bit.ly/3gkY7JX)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-Latvian | NMT            | Life Sciences| MonoTransQuest      | [model.zip](https://bit.ly/3eLb1jU)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-Czech   | SMT            |  IT          | MonoTransQuest      | [model.zip](https://bit.ly/2VyBOZ2)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| German-English  | SMT            | Life Sciences| MonoTransQuest      | [model.zip](https://bit.ly/3dRlqJu)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-Chinese | NMT            | Wikipedia    | MonoTransQuest      | [model.zip](https://bit.ly/2YLIvJw)  | 
|                 |                |              | SiameseTransQuest   |                                      | 
| English-*       | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 
| \*-\*           | Any            | Any          | MonoTransQuest      |                                      | 
|                 |                |              | SiameseTransQuest   |                                      | 

!!! note
\* denotes any language. (\*-\* means any language to any language)