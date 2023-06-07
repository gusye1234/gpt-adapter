<div align="center">
  <h1>gpt-adapter🫥</h1>
  <p><strong>Enable quick fine-tuning for popular HuggingFace🤗 models in one line</strong></p>
</div>



## Quick start

```python
from transformers import AutoModel
import gpt_adapter
model = AutoModel.from_pretrained("facebook/opt-350m)

model = gpt_adapter.add_adapter(model, adapter_name="opt_adapter")
```

After `gpt_adapter.add_adapter`, the most of the parameters of `model` will be frozen and the attention layer will be replace by the adapter layer, referring to [Llama-Adapter](https://arxiv.org/pdf/2303.16199.pdf).



## Install

```
git clone https://github.com/gusye1234/gpt-adapter.git
cd gpt-adapter
pip install -e .
```

## TODOs

More models to support

- [ ] LLama
- [ ] GPT
- [ ] BLOOM

More PEFT algorithms to support

- [ ] Lora
- [ ] P-tuning
- [ ] ...
