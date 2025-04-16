import subprocess
import youtokentome as yttm
from transformers import (EncoderDecoderModel, GPT2Config
                          , BertModel, GPT2LMHeadModel,)
import torch

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}


#Load tokenizer
# Don't retrain the tokenizer, just load the one used for training
print("[INFO] Loading tokenizer used during training...")
tokenizer = yttm.BPE(model="abc_tokenizer.model")  # This must match the training tokenizer

VOCAB_SIZE = tokenizer.vocab_size()
print("Tokenizer vocab size:", VOCAB_SIZE)

print("Special tokens in tokenizer:")
for i in range(10):
    print(f"ID {i} => {tokenizer.id_to_subword(i)}")

# Read your ABC file
with open("output.abc", "r") as f:
    abc_text = f.read()


# Tokenize it
input_ids = tokenizer.encode(abc_text, output_type=yttm.OutputType.ID)
print("Tokenized Input IDs:", input_ids)

max_id = max(input_ids)
print("Max token ID in input:", max_id)
print("Vocab size:", tokenizer.vocab_size())

# Ensure all token IDs are in range
if max(input_ids) >= VOCAB_SIZE:
    print("Truncating out-of-range token IDs")
    input_ids = [id for id in input_ids if id < VOCAB_SIZE]
assert max(input_ids) < VOCAB_SIZE, "Found token ID >= vocab size"

input_tensor = torch.tensor([input_ids], dtype=torch.long)
attention_mask = torch.ones_like(input_tensor)

# Step 1: Load and override GPT2 config manually
gpt2_config = GPT2Config.from_pretrained("./abc_transformer_model/decoder")
gpt2_config.vocab_size = VOCAB_SIZE
gpt2_config.pad_token_id = 0
gpt2_config.eos_token_id = 3
gpt2_config.bos_token_id = 2
gpt2_config.is_decoder = True
gpt2_config.add_cross_attention = True

# === Load encoder-decoder model
encoder = BertModel.from_pretrained("./abc_transformer_model/encoder")
decoder = GPT2LMHeadModel.from_pretrained("./abc_transformer_model/decoder", config=gpt2_config)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# === Resize decoder token embeddings
model.decoder.resize_token_embeddings(VOCAB_SIZE)
print("Vocab size in tokenizer:", tokenizer.vocab_size())
print("Decoder embedding size:", model.decoder.get_input_embeddings().weight.size(0))
max_id = max(input_ids)
embedding_size = model.decoder.transformer.wte.weight.shape[0]
print(f"Max token ID in input: {max_id}")
print(f"Decoder embedding size: {embedding_size}")

eos_token = 3  # </s>
pad_token = 0  # <pad>
eos_token = tokenizer.subword_to_id("<EOS>")
pad_token = tokenizer.subword_to_id("<PAD>")

print(f"EOS token ID: {eos_token}")
print(f"PAD token ID: {pad_token}")

assert eos_token == 3 and pad_token == 0, f"Unexpected token IDs: eos={eos_token}, pad={pad_token}"
assert eos_token is not None and pad_token is not None
assert max(input_ids) < VOCAB_SIZE

model.config.decoder_start_token_id = tokenizer.subword_to_id("<BOS>")
model.config.pad_token_id = pad_token
model.config.eos_token_id = eos_token

print("pad_token:", tokenizer.id_to_subword(0))
print("unk_token:", tokenizer.id_to_subword(1))
print("bos_token:", tokenizer.id_to_subword(2))
print("eos_token:", tokenizer.id_to_subword(3))

model.config.decoder_start_token_id = 2  # <s>
model.config.pad_token_id = 0            # <pad>
model.config.eos_token_id = 3            # </s>

print("model.config.pad_token_id:", model.config.pad_token_id)
print("model.config.eos_token_id:", model.config.eos_token_id)
print("model.config.decoder_start_token_id:", model.config.decoder_start_token_id)

output_ids = model.generate(
    input_ids=input_tensor,
    attention_mask=attention_mask,
    max_length=512,
    num_beams=5,
    no_repeat_ngram_size=3,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id
)[0].tolist()
# Decode the tokenized output
generated_abc = tokenizer.decode(output_ids)
print("Generated ABC:\n", generated_abc)

def clean_abc(generated_abc: str) -> str:
    cleaned = generated_abc.replace("]2", "]").replace("[", "").replace("]", "").replace("\\", "")
    cleaned = cleaned.replace(" | ", " |").replace("||", "|").strip()
    return cleaned

