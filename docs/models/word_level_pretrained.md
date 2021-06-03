# Word Level Pre-trained Models

We have released several pre-trained TransQuest models on word-level quality estimation and they are available on [Hugging Face Model Hub](). We will be keep releasing new models. So please keep in touch.

### Available Models

| Language Pair   | NMT/SMT        |  Domain      |     Algorithm        |  Model Name and Link to HUgging Face | 
|:---------------:|:--------------:|:------------:|:--------------------:|:------------------------------------:|
| English-German  | NMT            |  IT          | MicroTransQuest      | [TransQuest/microtransquest-en_de-it-nmt](https://huggingface.co/TransQuest/microtransquest-en_de-it-nmt)  |
|                 | NMT            |  Wiki        | MicroTransQuest      |                                     | 
|                 | SMT            |  IT          | MicroTransQuest      | [TransQuest/microtransquest-en_de-it-smt](https://huggingface.co/TransQuest/microtransquest-en_de-it-smt)  | 
| English-Latvian | NMT            | Life Sciences| MicroTransQuest      | [TransQuest/microtransquest-en_lv-pharmaceutical-nmt](https://huggingface.co/TransQuest/microtransquest-en_lv-pharmaceutical-nmt)| 
|                 | SMT            | Life Sciences| MicroTransQuest      | [TransQuest/microtransquest-en_lv-pharmaceutical-smt](https://huggingface.co/TransQuest/microtransquest-en_lv-pharmaceutical-smt)  | 
| English-Czech   | SMT            |  IT          | MicroTransQuest      | [TransQuest/microtransquest-en_cs-it-smt ](https://huggingface.co/TransQuest/microtransquest-en_cs-it-smt)  |  
| German-English  | SMT            | Life Sciences| MicroTransQuest      | [TransQuest/microtransquest-de_en-pharmaceutical-smt](https://huggingface.co/TransQuest/microtransquest-de_en-pharmaceutical-smt)  | 
| English-Chinese | NMT            | Wikipedia    | MicroTransQuest      |                                     | 
| English-Russian | NMT            | IT           | MicroTransQuest      |                                     | 
| \*-\*           | Any            | Any          | MicroTransQuest      |                                     | 

You can load any of the above models with the below code. The full notebook is available [here.](https://colab.research.google.com/drive/1fslfFoQnspdv2Do5hfwwkbEwow-CDIxD?usp=sharing)

```python
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
import torch

model = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_lv-pharmaceutical-nmt", labels=["OK", "BAD"], use_cuda=torch.cuda.is_available())
```




!!! note
    \* denotes any language. (\*-\* means any language to any language)