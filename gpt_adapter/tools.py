import torch
from .utils import get_module_device


class EasyChatBot:
    def __init__(self, model, tokenizer, template):
        self.template = template
        self.tokenizer = tokenizer
        self.model = model
        self.device = get_module_device(model)

    @torch.no_grad()
    def chat(self, input_str):
        input_str = self.template.format_map(
            {
                "instruction": input_str,
            }
        )
        input_ids = self.tokenizer.encode(input_str, return_tensors="pt").to(
            self.device
        )
        output = self.model.generate(input_ids, max_new_tokens=40)[0]
        output_str = self.tokenizer.decode(output, skip_special_tokens=True)

        return output_str[len(input_str) :]

    def console(self):
        while True:
            wait_input = input(">>")
            if len(wait_input) == 0:
                continue
            print(self.chat(wait_input))
            print("---------")
