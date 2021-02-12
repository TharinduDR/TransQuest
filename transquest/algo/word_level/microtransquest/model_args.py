from dataclasses import dataclass, field

from transquest.algo.sentence_level.monotransquest.model_args import TransQuestArgs


@dataclass
class MicroTransQuestArgs(TransQuestArgs):
    """
    Model args for a MicroTransQuestModel
    """

    model_class: str = "MicroTransQuestModel"
    classification_report: bool = False
    labels_list: list = field(default_factory=list)
    lazy_loading: bool = False
    lazy_loading_start_line: int = 0
    onnx: bool = False
    special_tokens_list: list = field(default_factory=list)
    add_tag: bool = False
    tag: str = "<gap>"
    default_quality: str = "OK"
