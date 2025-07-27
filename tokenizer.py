from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding

# Load your custom encoding or the one provided by OpenAI
encoding = get_encoding("gpt2")
convert_tiktoken_to_fast(encoding, "config/save/dir")