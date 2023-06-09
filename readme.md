<div align="center">
  <h1>gpt-adapter🫥</h1>
  <p><strong>Enable quick fine-tuning for popular HuggingFace🤗 models in one line</strong></p>
      <p>
    <a href="https://github.com/gusye1234/gpt-adapter/actions/workflows/main.yml">
      <img src="https://github.com/gusye1234/gpt-adapter/actions/workflows/main.yml/badge.svg">
    </a>
        <a href="https://codecov.io/gh/gusye1234/gpt-adapter" >
 <img src="https://codecov.io/gh/gusye1234/gpt-adapter/branch/main/graph/badge.svg?token=q4ajb6LVcm"/> </a>
 	</p>
</div>

## Examples
Few lines of clean code to load a ready-to-chat LLM: [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main):
```python
import gpt_adapter
from gpt_adapter import convert
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(<LLAMA_DIR>)
model = gpt_adapter.add_adapter(model, adapter_name="llama_adapter", start_num=2, adapter_len=10)

add_state_dict = convert.convert_llama_adapter_v1(<ADAPTER_WEIGHT>)
model.load_state_dict(add_state_dict, strict=False)
```
where `<LLAMA_DIR>` is your llama-7B checkpoint dir or unofficial huggingface models(*e.g.* `nyanko7/LLaMA-7B`), `<ADAPTER_WEIGHT>` is the adapters weight of LLaMA-Adapter which is really small and can be downloaded from [here](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main/alpaca_finetuning_v1).

A chat script is ready at `examples/load_llama_adapterv1.py`. Replace those paths and you can have a AI assistant:
```
>>Hi!
Hello!
...
```
## Quick start

```python
from transformers import AutoModel
import gpt_adapter
model = AutoModel.from_pretrained("facebook/opt-350m")

model = gpt_adapter.add_adapter(model, adapter_name="opt_adapter")
```

After `gpt_adapter.add_adapter`, the most of the parameters of `model` will be frozen and the attention layer will be replaced by the adapter layer, referring to [Llama-Adapter](https://arxiv.org/pdf/2303.16199.pdf).

By doing this, the trainable parameters are dramatically reduced. A six-layer OPT model with last 3 layer is adapted:
```
Normal trainable parameters                   Using gpt-adapter
----------------------------------------      -----------------------------------------------
decoder.embed_tokens.weight                   decoder.layers.3.self_attn.gate
decoder.embed_positions.weight                decoder.layers.3.self_attn.adapter_prefix.weight
decoder.final_layer_norm.weight               decoder.layers.4.self_attn.gate
decoder.final_layer_norm.bias                 decoder.layers.4.self_attn.adapter_prefix.weight
decoder.layers.0.self_attn.k_proj.weight      decoder.layers.5.self_attn.gate
decoder.layers.0.self_attn.k_proj.bias        decoder.layers.5.self_attn.adapter_prefix.weight
decoder.layers.0.self_attn.v_proj.weight
decoder.layers.0.self_attn.v_proj.bias
...
decoder.layers.5.fc1.weight
decoder.layers.5.fc1.bias
decoder.layers.5.fc2.weight
decoder.layers.5.fc2.bias
decoder.layers.5.final_layer_norm.weight
decoder.layers.5.final_layer_norm.bias
```

## Install

```
git clone https://github.com/gusye1234/gpt-adapter.git
cd gpt-adapter
pip install -e .
```

## TODOs

More models to support
- [x] OPT
- [x] LLama
- [ ] GPT
- [ ] BLOOM

More PEFT algorithms to support
- [x] Adapter
- [ ] Lora
- [ ] P-tuning
- [ ] ...
