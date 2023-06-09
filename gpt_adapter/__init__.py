import logging

logging.basicConfig(level=logging.INFO)

from .utils import *
from .adapter import *

__author__ = "Gus"
__url__ = "https://github.com/gusye1234/gpt-adapter"
__version__ = "0.1"

ADAPTERS = {
    "opt_adapter": (
        OPTNaiveAdapter,
        get_attention_type,
        opt_adapter_set_kwargs,
        normal_transit_fn,
    ),
    "llama_adapter": (
        LlamaNaiveAdapter,
        get_attention_type,
        llama_adapter_set_kwargs,
        normal_transit_fn,
    ),
}


def add_adapter(
    model,
    adapter_name=None,
    adapter_module=None,
    replace_module_fn=None,
    set_kwargs_fn=None,
    transit_fn=None,
    **kwargs
):
    if adapter_name is not None:
        template = ADAPTERS[adapter_name]
        adapter_module = adapter_module or template[0]
        replace_module_fn = replace_module_fn or template[1]
        set_kwargs_fn = set_kwargs_fn or template[2]
        transit_fn = transit_fn or template[3]
    start_num = kwargs.pop("start_num", 0)

    replace_module = replace_module_fn(model)
    model = locate_replace_module(
        model,
        replace_module,
        adapter_module,
        start_num=start_num,
        set_kwargs_fn=set_kwargs_fn,
        transit_fn=transit_fn,
        **kwargs
    )
    model.config.adapter_start_num = start_num
    return model
