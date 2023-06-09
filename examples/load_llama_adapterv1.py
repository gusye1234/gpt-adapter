import torch
import sys

sys.path.append("../")
import gpt_adapter
from gpt_adapter import convert, tools
from transformers import AutoModelForCausalLM, LlamaTokenizer

TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
LLAMA_DIR = "/data/mm64/jianbaiye/workspace/fixtures/llama-7b-hf"
ADAPTER_WEIGHT = (
    "/data/mm64/jianbaiye/workspace/fixtures/llama_adapter_len10_layer30_release.pth"
)

model = AutoModelForCausalLM.from_pretrained(LLAMA_DIR, torch_dtype=torch.float16)
tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)
model = gpt_adapter.add_adapter(
    model, adapter_name="llama_adapter", start_num=2, adapter_len=10
)


add_state_dict = convert.convert_llama_adapter_v1(ADAPTER_WEIGHT)
model.load_state_dict(add_state_dict, strict=False)

if torch.cuda.is_available():
    model = model.cuda()
else:
    # some CPU operators don't support float16
    model = model.float()
bot = tools.EasyChatBot(model, tokenizer, TEMPLATE)
bot.console()
