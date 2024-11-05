import mindspore
from mindnlp.transformers import MiniCPMForCausalLM, AutoTokenizer, StaticCache
from mindnlp.core import ops
from mindnlp.configs import set_pyboost
from mindnlp.quant.smooth_quant import quantize, w8x8
import time


NUM_TOKENS_TO_GENERATE = 40

model_id = "OpenBMB/MiniCPM-2B-dpo-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_id, mirror="modelscope")
model = MiniCPMForCausalLM.from_pretrained(model_id, ms_dtype=mindspore.float16, mirror="modelscope")

# quantize_cfg = w8x8(model.model.config)
# quantize(model, cfg=quantize_cfg)

# model.jit()

# prompts = [
#     "北京是中国的首都。这个城市 "
# ]
# model_inputs = tokenizer(prompts, return_tensors="ms")

messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="ms", add_generation_prompt=True, return_dict=True)
# set_pyboost(False)

# @mindspore.jit(jit_config=mindspore.JitConfig(jit_syntax_level='STRICT'))
def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True,
        use_static_cache=True,
    )[0]
    new_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token

batch_size, seq_length = model_inputs["input_ids"].shape
past_key_values = StaticCache(
    config=model.config, max_batch_size=1, max_cache_len=512, dtype=model.dtype
)
cache_position = ops.arange(seq_length)
generated_ids = ops.zeros(
    batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=mindspore.int32
)
generated_ids[:, cache_position] = model_inputs["input_ids"].to(mindspore.int32)

logits = model(
    **model_inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True, use_static_cache=True
)[0]
next_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
generated_ids[:, seq_length] = next_token[:, 0]

cache_position = mindspore.tensor([seq_length + 1])
for _ in range(1, NUM_TOKENS_TO_GENERATE):
    s = time.time()
    next_token = decode_one_tokens(model, next_token, None, cache_position, past_key_values)
    t = time.time()
    print(t - s)
    generated_ids[:, cache_position] = next_token.int()
    cache_position += 1

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(text)