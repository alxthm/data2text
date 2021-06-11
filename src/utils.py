from collections import OrderedDict
from pathlib import Path
from typing import Union, List, Tuple, Dict

import mlflow
import numpy as np
import random

import torch
from torch import nn


class WarningsFilter:
    def __init__(self, stream):
        """
        Filter some repetitive warnings to keep a clean stdout

        Args:
            stream: can be sys.stdout or sys.stderr for instance
        """
        self.stream = stream
        # 21/05/20 20:19:06 WARN hdfs.DFSUtil: Namenode for yarn-experimental
        # remains unresolved for ID 1.  Check your hdfs-site.xml file to ensure
        # namenodes are configured properly.
        self.strings_to_filter = [
            "Check your hdfs-site.xml file",
            "pyarrow.hdfs.connect is deprecated",
        ]

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        # check there is no forbidden string in the output data
        if all(s not in data for s in self.strings_to_filter):
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        self.stream.flush()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mlflow_log_src_and_config(conf, project_dir: Path):
    # log hyperparameters in config
    for k1 in conf.keys():
        if hasattr(conf[k1], "keys"):
            for k2 in conf[k1].keys():
                mlflow.log_param(f"{k1}.{k2}", conf[k1][k2])
        else:
            mlflow.log_param(f"{k1}", conf[k1])

    # save source files
    for f in (project_dir / "src").rglob("*.py"):
        mlflow.log_artifact(str(f), f"code/{f.relative_to(project_dir)}")


def camel_case_to_natural_text(s: str):
    # https://stackoverflow.com/a/44969381
    return "".join([" " + c.lower() if c.isupper() else c for c in s]).lstrip(" ")


def get_precision_recall_f1(num_correct, num_predicted, num_gt):
    assert 0 <= num_correct <= num_predicted
    assert 0 <= num_correct <= num_gt

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_gt if num_gt > 0 else 0.0
    f1 = 2.0 / (1.0 / precision + 1.0 / recall) if num_correct > 0 else 0.0

    return precision, recall, f1


# from https://github.com/PyTorchLightning/pytorch-lightning/


class LayerSummary(object):
    """
    Summary class for a single layer in a :class:`~pytorch_lightning.core.lightning.LightningModule`.
    It collects the following information:
    - Type of the layer (e.g. Linear, BatchNorm1d, ...)
    - Input shape
    - Output shape
    - Number of parameters
    The input and output shapes are only known after the example input array was
    passed through the model.
    Example::
        >>> model = torch.nn.Conv2d(3, 8, 3)
        >>> summary = LayerSummary(model)
        >>> summary.num_parameters
        224
        >>> summary.layer_type
        'Conv2d'
        >>> output = model(torch.rand(1, 3, 5, 5))
        >>> summary.in_size
        [1, 3, 5, 5]
        >>> summary.out_size
        [1, 8, 3, 3]
    Args:
        module: A module to summarize
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self._module = module
        self._in_size = None
        self._out_size = None

    @property
    def in_size(self) -> Union[str, List]:
        return self._in_size or "?"

    @property
    def out_size(self) -> Union[str, List]:
        return self._out_size or "?"

    @property
    def layer_type(self) -> str:
        """Returns the class name of the module."""
        return str(self._module.__class__.__name__)

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(np.prod(p.shape) for p in self._module.parameters())


class ModelSummary(object):
    """
    Generates a summary of all layers in a :class:`~pytorch_lightning.core.lightning.LightningModule`.
    Args:
        model: The model to summarize (also referred to as the root module)
        mode: Can be one of
             - `top` (default): only the top-level modules will be recorded (the children of the root module)
             - `full`: summarizes all layers and their submodules in the root module
    The string representation of this summary prints a table with columns containing
    the name, type and number of parameters for each layer.
    The root module may also have an attribute ``example_input_array`` as shown in the example below.
    If present, the root module will be called with it as input to determine the
    intermediate input- and output shapes of all layers. Supported are tensors and
    nested lists and tuples of tensors. All other types of inputs will be skipped and show as `?`
    in the summary table. The summary will also display `?` for layers not used in the forward pass.
    Example::
        >>> import pytorch_lightning as pl
        >>> class LitModel(pl.LightningModule):
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512))
        ...         self.example_input_array = torch.zeros(10, 256)  # optional
        ...
        ...     def forward(self, x):
        ...         return self.net(x)
        ...
        >>> model = LitModel()
        >>> ModelSummary(model, mode='top')  # doctest: +NORMALIZE_WHITESPACE
          | Name | Type       | Params | In sizes  | Out sizes
        ------------------------------------------------------------
        0 | net  | Sequential | 132 K  | [10, 256] | [10, 512]
        ------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
        >>> ModelSummary(model, mode='full')  # doctest: +NORMALIZE_WHITESPACE
          | Name  | Type        | Params | In sizes  | Out sizes
        --------------------------------------------------------------
        0 | net   | Sequential  | 132 K  | [10, 256] | [10, 512]
        1 | net.0 | Linear      | 131 K  | [10, 256] | [10, 512]
        2 | net.1 | BatchNorm1d | 1.0 K    | [10, 512] | [10, 512]
        --------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
    """

    def __init__(self, model, mode: str):
        self._model = model
        self._mode = mode
        self._layer_summary = self.summarize()

    @property
    def named_modules(self) -> List[Tuple[str, nn.Module]]:
        if self._mode == "full":
            mods = self._model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        elif self._mode == "top":
            # the children are the top-level modules
            mods = self._model.named_children()
        else:
            mods = []
        return list(mods)

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> List[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> List[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def summarize(self) -> Dict[str, LayerSummary]:
        summary = OrderedDict(
            (name, LayerSummary(module)) for name, module in self.named_modules
        )
        return summary

    def __str__(self):
        """
        Makes a summary listing with:
        Layer Name, Layer Type, Number of Parameters
        """
        arrays = [
            [" ", list(map(str, range(len(self._layer_summary))))],
            ["Name", self.layer_names],
            ["Type", self.layer_types],
            ["Params", list(map(get_human_readable_count, self.param_nums))],
        ]
        total_parameters = self.total_parameters
        trainable_parameters = self.trainable_parameters

        return _format_summary_table(total_parameters, trainable_parameters, *arrays)

    def __repr__(self):
        return str(self)


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


def _format_summary_table(
    total_parameters: int, trainable_parameters: int, *cols
) -> str:
    """
    Takes in a number of arrays, each specifying a column in
    the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted.
    """
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width

    summary += "\n" + s.format(get_human_readable_count(trainable_parameters), 10)
    summary += "Trainable params"
    summary += "\n" + s.format(
        get_human_readable_count(total_parameters - trainable_parameters), 10
    )
    summary += "Non-trainable params"
    summary += "\n" + s.format(get_human_readable_count(total_parameters), 10)
    summary += "Total params"

    return summary
