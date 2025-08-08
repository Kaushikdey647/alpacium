# FinSeer Model Card

![overview](./Overview.png)

## Model details

**Model type:**
StockLLM is an open-source fine-tuned 1B large language model as the backbone of our first retrieval-augmented generation (RAG) framework specifically designed for financial time-series forecasting.

**Paper or resources for more information:**

https://arxiv.org/pdf/2502.05878


## Use a pipeline as a high-level helper

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="TheFinAI/StockLLM")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
pipe(text=messages)
```

## Load model directly

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheFinAI/StockLLM")
model = AutoModelForCausalLM.from_pretrained("TheFinAI/StockLLM")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)   
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

## Base Model

LlamaForCausalLM
class transformers.LlamaForCausalLM
<
source
>
( config )

Parameters

config (LlamaForCausalLM) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the from_pretrained() method to load the model weights.
The Llama Model for causal language modeling.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

forward
<
source
>
( input_ids: typing.Optional[torch.LongTensor] = Noneattention_mask: typing.Optional[torch.Tensor] = Noneposition_ids: typing.Optional[torch.LongTensor] = Nonepast_key_values: typing.Optional[transformers.cache_utils.Cache] = Noneinputs_embeds: typing.Optional[torch.FloatTensor] = Nonelabels: typing.Optional[torch.LongTensor] = Noneuse_cache: typing.Optional[bool] = Nonecache_position: typing.Optional[torch.LongTensor] = Nonelogits_to_keep: typing.Union[int, torch.Tensor] = 0**kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

Parameters

input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.
Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are input IDs?

attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional) — Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
1 for tokens that are not masked,
0 for tokens that are masked.
What are attention masks?

position_ids (torch.LongTensor of shape (batch_size, sequence_length), optional) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range [0, config.n_positions - 1].
What are position IDs?

past_key_values (~cache_utils.Cache, optional) — Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the past_key_values returned by the model at a previous stage of decoding, when use_cache=True or config.use_cache=True.
Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional) — Optionally, instead of passing input_ids you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert input_ids indices into associated vectors than the model’s internal embedding lookup matrix.
labels (torch.LongTensor of shape (batch_size, sequence_length), optional) — Labels for computing the masked language modeling loss. Indices should either be in [0, ..., config.vocab_size] or -100 (see input_ids docstring). Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size].
use_cache (bool, optional) — If set to True, past_key_values key value states are returned and can be used to speed up decoding (see past_key_values).
cache_position (torch.LongTensor of shape (sequence_length), optional) — Indices depicting the position of the input sequence tokens in the sequence. Contrarily to position_ids, this tensor is not affected by padding. It is used to update the cache in the correct position and to infer the complete sequence length.
logits_to_keep (Union[int, torch.Tensor], defaults to 0) — If an int, compute logits for the last logits_to_keep tokens. If 0, calculate logits for all input_ids (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a torch.Tensor, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).
Returns

transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (LlamaConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction).

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


The LlamaForCausalLM forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Copied
from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."