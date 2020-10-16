from dataclasses import dataclass, field

from transquest.algo.transformers.model_args import ModelArgs


@dataclass
class MiniTransQuestArgs(ModelArgs):
    """
    Model args for a MiniTransQuestModel
    """

    model_class: str = "MiniTransQuestModel"
    classification_report: bool = False
    labels_list: list = field(default_factory=list)
    lazy_loading: bool = False
    lazy_loading_start_line: int = 0
    onnx: bool = False