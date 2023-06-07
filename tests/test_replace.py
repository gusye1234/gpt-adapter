import torch
import pytest
import gpt_adapter
from transformers import AutoModel
from transformers.models.opt.modeling_opt import OPTAttention

def test_opt_llama_adapter():
    model = AutoModel.from_pretrained("facebook/opt-125m")
    mock_input = torch.randint(0, 1000, (3, 20), dtype=torch.long)
    if torch.cuda.is_available():
        model = model.cuda()
        mock_input = mock_input.cuda()

    assert gpt_adapter.get_attention_type(model) == OPTAttention

    print(model)
    model = gpt_adapter.add_adapter(model, adapter_name="llama_adapter", adapter_len=5, start_num=5)
    print(model)
    print(model(mock_input)["last_hidden_state"].shape)