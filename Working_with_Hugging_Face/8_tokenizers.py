# Import the AutoTokenizer
from transformers import AutoTokenizer, GPT2Tokenizer, DistilBertTokenizer

input_string="HOWDY, how aré yoü?"

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Normalize the input string
output = tokenizer.backend_tokenizer.normalizer.normalize_str(input_string)


print(f"distilbert-base-uncased normalize result : {output}")



################################ Comparing 2 Tokenizer #####################################

# Download the gpt tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the input
gpt_tokens = gpt_tokenizer.tokenize(input_string)

# Repeat for distilbert
distil_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
distil_tokens = distil_tokenizer.tokenize(text=input_string)

# Compare the output
print(f"distilbert-base-uncased tokenizer: {tokenizer.tokenize(input_string)}")
print(f"GPT tokenizer: {gpt_tokens}")
print(f"DistilBERT tokenizer: {distil_tokens}")