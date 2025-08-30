from vllm import LLM

llm = LLM(
    model="/pfs/work9/workspace/scratch/ul_swv79-pixtral/pixtral-12b/",
    tokenizer_mode="mistral"
)

tokenizer = llm.tokenizer
print(f"Tokenizer type: {type(tokenizer)}")
print(f"Available methods: {dir(tokenizer)}")

token = "<image>"

if hasattr(tokenizer, "convert_tokens_to_ids"):
    print("convert_tokens_to_ids found:")
    print(tokenizer.convert_tokens_to_ids(token))
elif hasattr(tokenizer, "token_to_id"):
    print("token_to_id found:")
    print(tokenizer.token_to_id(token))
else:
    print("No suitable method found to get token id.")
