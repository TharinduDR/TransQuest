from examples.wmt_2020.common.util.download import download_from_google_drive
from examples.wmt_2020.ro_en.transformer_nmt_config import MODEL_TYPE, transformer_nmt_config, DRIVE_FILE_ID, \
    MODEL_NAME, GOOGLE_DRIVE, TEMP_DIRECTORY, RESULT_FILE
from transquest.algo.transformers.run_model import QuestModel
import torch
import tarfile
import urllib.request
import os

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

urllib.request.urlretrieve ("https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/training_ro-en.tar.gz", "training_ro-en.tar.gz")


model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=transformer_nmt_config)

tar = tarfile.open("training_ro-en.tar.gz", "r:gz")
tar.extractall()
tar.close()

with open('train.roen.ro') as f:
    romanian_lines = f.read().splitlines()

with open('train.roen.en') as f:
    english_lines = f.read().splitlines()

nmt_sentence_pairs = list(map(list, zip(romanian_lines, english_lines)))
predictions, raw_outputs = model.predict(nmt_sentence_pairs)

with open(os.path.join(TEMP_DIRECTORY, RESULT_FILE), "w") as f:
    for s in predictions:
        f.write(str(s) +"\n")