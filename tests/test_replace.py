import torch
import pytest
import gpt_adapter
from tabulate import tabulate
from gpt_adapter import if_field_in_key
from transformers import AutoModel
from transformers.models.opt.modeling_opt import OPTAttention, OPTModel, OPTConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaModel,
)


def assert_grad_parameters(names, module):
    for key, param in module.named_parameters():
        if any([if_field_in_key(n, key) for n in names]):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_opt_adapter():
    opt_config = OPTConfig(hidden_size=12 * 6, num_hidden_layers=6)
    model = OPTModel(opt_config)

    mock_input = torch.randint(0, 1000, (1, 4), dtype=torch.long)
    if torch.cuda.is_available():
        model = model.cuda()
        mock_input = mock_input.cuda()

    assert gpt_adapter.get_attention_type(model) == OPTAttention

    before_str = "\n".join([k for k, p in model.named_parameters() if p.requires_grad])
    model = gpt_adapter.add_adapter(
        model, adapter_name="opt_adapter", adapter_len=5, start_num=3
    )
    after_str = "\n".join([k for k, p in model.named_parameters() if p.requires_grad])
    print(tabulate([[before_str, after_str]]))
    print(model(mock_input)["last_hidden_state"].shape)
    assert_grad_parameters(["gate", "adapte_prefix"], model)


def test_llama_adapter():
    llama_config = LlamaConfig(hidden_size=64, num_hidden_layers=6)
    model = LlamaModel(llama_config)

    mock_input = torch.randint(0, 1000, (1, 4), dtype=torch.long)
    if torch.cuda.is_available():
        model = model.cuda()
        mock_input = mock_input.cuda()

    assert gpt_adapter.get_attention_type(model) == LlamaAttention

    print(model)
    model = gpt_adapter.add_adapter(
        model, adapter_name="llama_adapter", adapter_len=5, start_num=3
    )
    print(model)
    print(model(mock_input)["last_hidden_state"].shape)
    assert_grad_parameters(["gate", "adapte_prefix"], model)
