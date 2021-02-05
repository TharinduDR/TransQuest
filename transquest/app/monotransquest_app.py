import logging
import os

from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from transquest.app.util.model_downloader import GoogleDriveDownloader as gdd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MonoTransQuestApp:
    def __init__(self, model_name_or_path, model_type=None, use_cuda=True, force_download=False, cuda_device=-1):

        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device

        MODEL_CONFIG = {
            "monotransquest-da-si_en": ("xlmroberta", "1-UXvna_RGnb6_TTRr4vSGCqA5yl0SYn9"),
            "monotransquest-da-ro_en": ("xlmroberta", "1-aeDbR_ftqsTslFJbNybebj5MAhPfIw8")
        }

        if model_name_or_path in MODEL_CONFIG:
            self.trained_model_type, self.drive_id = MODEL_CONFIG[model_name_or_path]

            try:
                from torch.hub import _get_torch_home
                torch_cache_home = _get_torch_home()
            except ImportError:
                torch_cache_home = os.path.expanduser(
                    os.getenv('TORCH_HOME', os.path.join(
                        os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
            default_cache_path = os.path.join(torch_cache_home, 'transquest')
            self.model_path = os.path.join(default_cache_path, self.model_name_or_path)
            if force_download or (not os.path.exists(self.model_path) or not os.listdir(self.model_path)):
                print(
                    "Downloading a MonoTransQuest model and saving it at {}".format(self.model_path))

                gdd.download_file_from_google_drive(file_id=self.drive_id,
                                                    dest_path=os.path.join(self.model_path, "model.zip"),
                                                    showsize=True, unzip=True)

            self.model = MonoTransQuestModel(self.trained_model_type, self.model_path, use_cuda=self.use_cuda,
                                             cuda_device=self.cuda_device)

        else:
            self.model = MonoTransQuestModel(model_type, self.model_name_or_path, use_cuda=self.use_cuda,
                                             cuda_device=self.cuda_device)

    @staticmethod
    def _download(drive_id, model_name):
        gdd.download_file_from_google_drive(file_id=drive_id,
                                            dest_path=os.path.join(".transquest", model_name, "model.zip"),
                                            unzip=True)

    def predict_quality(self, test_sentence_pairs):
        predictions, raw_outputs = self.model.predict(test_sentence_pairs)
        return predictions
