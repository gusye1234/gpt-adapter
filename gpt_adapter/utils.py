import torch
from torch import nn
import logging


def get_module_device(module: nn.Module):
    return next(module.parameters()).device


def get_hidden_size(model):
    if hasattr(model, "config"):
        return getattr(model.config, "hidden_size")
    raise NotImplementedError(
        "Only support huggingface model for now"
        ", or you should set your model configure in `model.config`"
    )


def get_model_layers(model):
    if hasattr(model, "config"):
        return getattr(model.config, "num_hidden_layers")
    raise NotImplementedError(
        "Only support huggingface model for now"
        ", or you should set your model configure in `model.config`"
    )


def get_attention_type(model):
    """TODO: very immature"""
    possible_types = []
    for key, module in model.named_modules():
        if "Attention" in str(type(module)):
            possible_types.append(type(module))
    possible_types = set(possible_types)
    if len(possible_types) == 1:
        return list(possible_types)[0]
    raise ValueError(f"More than one possible attention layer types! {possible_types}")


def normal_transit_fn(before_module: nn.Module, after_module: nn.Module):
    after_module = after_module.to(get_module_device(before_module))
    return after_module.load_state_dict(before_module.state_dict(), strict=False)


def replace_submodule(model, key, after):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    setattr(parent, target_name, after)
    return model


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def locate_replace_module(
    model: nn.Module,
    module_type,
    new_module_type,
    start_num=0,
    set_kwargs_fn=None,
    transit_fn=normal_transit_fn,
    **kwargs,
):
    count_time = 0
    for key, before_module in model.named_modules():
        freeze_module(before_module)
        before_module.eval()
        if not isinstance(before_module, module_type):
            continue
        count_time += 1
        if count_time <= start_num:
            continue

        # TODO: Double-copy module
        if set_kwargs_fn is None:
            after_module = new_module_type()
        else:
            kwargs = set_kwargs_fn(before_module, **kwargs)
            after_module = new_module_type(**kwargs)

        transit_fn(before_module, after_module)
        replace_submodule(model, key, after_module)

        logging.info(
            f"Replace one {key}, seen {count_time} times, {before_module} -> {after_module}"
        )
    return model
