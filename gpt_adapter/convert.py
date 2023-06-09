import torch
from .utils import if_field_in_key, get_hidden_size, get_model_layers


def convert_llama_adapter_v1(
    local_file,
    adapter_layer=30,
    adapter_len=10,
    total_layers=32,
    hidden_size=4096,
    dtype=torch.float16,
):
    add_weights = torch.load(local_file, map_location="cpu")
    add_weights = {k: v.to(dtype) for k, v in add_weights.items()}

    def trans_gate(before: str):
        fields = before.split(".")
        assert (
            len(fields) == 4
        ), "Unexpected key format, you may use the wrong converter"
        assert (
            fields[-1] == "gate"
        ), "Unexpected key format, you may use the wrong converter"
        new_name = ".".join(["model", fields[0], fields[1], "self_attn", "gate"])
        return new_name

    gpt_adapter_dict = {}
    for key, param in add_weights.items():
        if if_field_in_key("gate", key):
            gpt_adapter_dict[trans_gate(key)] = param.squeeze(dim=0)

    adapter_mat = add_weights["adapter_query.weight"]
    layer_len_num = adapter_mat.shape[0]
    assert (
        layer_len_num % adapter_layer == 0
    ), f"Expect the dim-0 of adapter_query.weight can be divided by num of adapter layers[{adapter_layer}]"

    prefix_len = layer_len_num // adapter_layer
    adapter_mat = adapter_mat.view(adapter_layer, prefix_len, -1)
    assert (
        adapter_mat.shape[-1] == hidden_size
    ), f"Expect the adapter query's hidden size is {hidden_size}, but got {adapter_mat.shape[-1]}"
    offset = total_layers - adapter_layer
    for layer_id in range(offset, total_layers):
        name = ".".join(
            ["model", "layers", str(layer_id), "self_attn", "adapter_prefix", "weight"]
        )
        gpt_adapter_dict[name] = adapter_mat[layer_id - offset]
    return gpt_adapter_dict
