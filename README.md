---
license: afl-3.0
language: uk
---
## GPT2 trained to generate ЗНО (Ukrainian exam SAT type of thing) essays

Generated texts are not very cohesive yet but I'm working on it. <br />
Use the code from the example below. The model takes "ZNOTITLE: your essay title" inputs. 


### Example of usage:
```python
from transformers import AlbertTokenizer, GPT2LMHeadModel
tokenizer = AlbertTokenizer.from_pretrained("kyryl0s/gpt2-uk-zno-edition")
model = GPT2LMHeadModel.from_pretrained("kyryl0s/gpt2-uk-zno-edition")
input_ids = tokenizer.encode("ZNOTITLE: За яку працю треба більше поважати людину - за фізичну чи інтелектуальну?", add_special_tokens=False, return_tensors='pt')
outputs = model.generate(
    input_ids,
    do_sample=True,
    num_return_sequences=1,
    max_length=250
)
for i, out in enumerate(outputs):
    print("{}: {}".format(i, tokenizer.decode(out)))
    
```
