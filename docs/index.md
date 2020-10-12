# TransQuest: Translation Quality Estimation with Cross-lingual Transformers
The goal of quality estimation (QE) is to evaluate the quality of a translation without having access to a reference translation. High-accuracy QE that can be easily deployed for a number of language pairs is the missing piece in many commercial translation workflows as they have numerous potential uses. They can be employed to select the best translation when several translation engines are available or can inform the end user about the reliability of automatically translated content. In addition, QE systems can be used to decide whether a translation can be published as it is in a given context, or whether it requires human post-editing before publishing or translation from scratch by a human. The quality estimation can be done at different levels: document level, sentence level and word level.

With TransQuest, we have opensourced our research in sentence-level quality estimation which also won the sentence-level direct assessment quality estimation shared task in [WMT 2020](http://www.statmt.org/wmt20/quality-estimation-task.html). TransQuest outperforms current open-source quality estimation frameworks such as [OpenKiwi](https://github.com/Unbabel/OpenKiwi) and [DeepQuest](https://github.com/sheffieldnlp/deepQuest).


## Installation
You first need to install PyTorch. The recommended PyTorch version is 1.5.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install TransQuest from source or from pip. We recommend installing form pip since it is stable.


### From pip

```bash
pip install transquest
```

### From Source

```bash
git clone https://github.com/TharinduDR/TransQuest.git
cd TransQuest
pip install -r requirements.txt
```




## Citation
If you are using the package, please consider citing this paper which is accepted to [COLING 2020](https://coling2020.org/)

```bash
@InProceedings{transquest:2020a,
author = {Ranasinghe, Tharindu and Orasan, Constantin and Mitkov, Ruslan},
title = {TransQuest: Translation Quality Estimation with Cross-lingual Transformers},
booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
year = {2020}
}
```

If you are using the task specific fine tuning, please consider citing this which is accepted to [WMT 2020](http://www.statmt.org/wmt20/) at EMNLP 2020.
 
```bash
@InProceedings{transquest:2020b,
author = {Ranasinghe, Tharindu and Orasan, Constantin and Mitkov, Ruslan},
title = {TransQuest at WMT2020: Sentence-Level Direct Assessment},
booktitle = {Proceedings of the Fifth Conference on Machine Translation},
year = {2020}
}
```


