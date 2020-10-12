# TransQuest Architectures
We have introduced two architectures in the TransQuest framework, both relies on the XLM-R transformer model.

##MonoTransQuest

The first architecture proposed uses a single XLM-R transformer model. The input of this model is a concatenation of the original sentence and its translation, separated by the *[SEP]* token. 

![MonoTransQuest Architecture](images/TransQuest.png)

### Minimal Start for a MonoTransQuest Model



##SiameseTransQuest 

![SiameseTransQuest Architecture](images/SiameseTransQuest.png)

### Minimal Start for a SiameseTransQuest Model