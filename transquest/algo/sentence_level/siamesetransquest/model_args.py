from dataclasses import dataclass, field

from transquest.model_args import TransQuestArgs


@dataclass
class SiameseTransQuestArgs(TransQuestArgs):
    """
    Model args for a SiameseTransQuest
    """

    model_class: str = "SiameseTransQuestModel"
    labels_list: list = field(default_factory=list)
    labels_map: dict = field(default_factory=dict)
    lazy_delimiter: str = "\t"
    lazy_labels_column: int = 1
    lazy_loading: bool = False
    lazy_loading_start_line: int = 1
    lazy_text_a_column: bool = None
    lazy_text_b_column: bool = None
    lazy_text_column: int = 0
    onnx: bool = False
    regression: bool = True
    sliding_window: bool = False
    special_tokens_list: list = field(default_factory=list)
    stride: float = 0.8
    tie_value: int = 1