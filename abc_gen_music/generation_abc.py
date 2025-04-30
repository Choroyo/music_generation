import youtokentome as yttm
from transformers import (EncoderDecoderModel, GPT2Config
                          , BertModel, GPT2LMHeadModel,)
import torch
from parameter import (TOKENIZER_MODEL3000, TOKENIZER_MODEL1200, MODEL_DIR3000,
                       MODEL_DIR1200, MODEL_DIR600,TOKENIZER_MODEL600, DATASET1200,
                       DATASET3000, DATASET600, VOCAB_SIZE3000, VOCAB_SIZE1200,
                       VOCAB_SIZE600)

import random
import re

def clean_abc(generated_abc, output_file):
    print("Cleaning ABC...")

    # 1. Strip <BOS>, <EOS>, and similar model tokens
    cleaned = re.sub(r"<BOS>|<EOS>|<PAD>|<UNK>", "", generated_abc)

    # 2. Fix malformed durations and token artifacts
    cleaned = re.sub(r"/2-+", "/2", cleaned)  # Fix "/2-"
    cleaned = re.sub(r"-+/2", "/2", cleaned)  # Fix "-/2"
    cleaned = re.sub(r"/2(?=\s|$)", "", cleaned)  # Remove isolated /2 at end of line
    cleaned = re.sub(r"[^\x20-\x7E\n]", "", cleaned)  # Remove non-printables

    # 3. Remove backslashes
    cleaned = cleaned.replace("\\", "")

    # 4. Collapse nested or extra brackets
    cleaned = re.sub(r"\[([^\]]*\[)+", "[", cleaned)
    cleaned = re.sub(r"\](\]*)+", "]", cleaned)

    # 5. Remove overly long numeric artifacts like "1666"
    cleaned = re.sub(r"\b\d{4,}\b", "", cleaned)  # Remove long standalone numbers

    # 6. Fix broken brackets (ensure count match)
    open_brackets = cleaned.count("[")
    close_brackets = cleaned.count("]")
    if open_brackets > close_brackets:
        cleaned += "]" * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        cleaned = "[" * (close_brackets - open_brackets) + cleaned

    # 7. Fix redundant whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove z or r from chords like [CzG]
    cleaned = re.sub(r"\[([^\]]*?)[zr]([^\]]*?)\]", r"[\1\2]", cleaned)

    # Remove invalid characters or artifacts inside chords
    cleaned = re.sub(r"\[.*?[^A-Ga-gz',/=^_0-9].*?\]", lambda m: re.sub(r'[^A-Ga-g\',/=^_0-9]', '', m.group(0)), cleaned)

    # Remove duplicate or malformed chord closures
    cleaned = re.sub(r"\[+|\]+", lambda m: '[' if '[' in m.group(0) else ']', cleaned)

    # 8. Save cleaned ABC
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(cleaned)

    print("Cleaned ABC:\n", cleaned)
    print(f"Saved cleaned ABC data to {output_file}")
    return cleaned

def wrap_abc_body_with_header(abc_body, output_file):
    meters = ["4/4", "6/8", "3/4"]
    keys = ["C", "D", "G", "F", "A", "E", "Bb", "Dm", "Am", "Em"]
    tempos = [60, 90, 120, 150]

    header = (
        "X:1\n"
        f"T:Generated Tune\n"
        f"M:{random.choice(meters)}\n"
        "L:1/8\n"
        f"Q:1/4={random.choice(tempos)}\n"
        f"K:{random.choice(keys)}\n"
        "%%MIDI program 0\n"
    )
    wrapped = header + abc_body.strip() + "\n"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(wrapped)
    print(f"Full ABC file saved to: {output_file}")
    return wrapped


def generate_abc(input_file, output_file, tokenizer_model, model_dir, vocab_size):
    #Load tokenizer
    print("Loading tokenizer used during training...")
    tokenizer = yttm.BPE(model=tokenizer_model)  # This must match the training tokenizer
    print("Tokenizer vocab size:", vocab_size)

    # Read your ABC file
    with open(input_file, "r") as f:
        abc_text = f.read()

    # Tokenize it
    input_ids = tokenizer.encode(abc_text, output_type=yttm.OutputType.ID)
    if len(input_ids) > 512:
        input_ids = input_ids[:512]
    print("Tokenized vocab size for input ids:", len(input_ids))
    print("Tokenized Input IDs:", input_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_tensor)

    # Step 1: Load and override GPT2 config manually
    gpt2_config = GPT2Config.from_pretrained(model_dir + "/decoder")
    gpt2_config.vocab_size = vocab_size
    gpt2_config.pad_token_id = 0
    gpt2_config.eos_token_id = 3
    gpt2_config.bos_token_id = 2
    gpt2_config.is_decoder = True
    gpt2_config.add_cross_attention = True

    # === Load encoder-decoder model
    encoder = BertModel.from_pretrained(model_dir + "/encoder")
    decoder = GPT2LMHeadModel.from_pretrained(model_dir + "/decoder", config=gpt2_config)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # === Resize decoder token embeddings
    #model.decoder.resize_token_embeddings(VOCAB_SIZE)

    eos_token = tokenizer.subword_to_id("<EOS>")
    pad_token = tokenizer.subword_to_id("<PAD>")
    model.config.decoder_start_token_id = tokenizer.subword_to_id("<BOS>")

    model.config.pad_token_id = pad_token
    model.config.eos_token_id = eos_token

    model.config.decoder_start_token_id = 2  # <s>
    model.config.pad_token_id = 0            # <pad>
    model.config.eos_token_id = 3            # </s>

    output_ids = model.generate(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=3,
        temperature=0.8,
        do_sample=True,
        top_k=20,
        top_p=0.8,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id
    )[0].tolist()
    # Decode the tokenized output
    generated_abc = tokenizer.decode(output_ids)
    print("Generated ABC:\n", generated_abc)
    print("Cleaning ABC...")
    if isinstance(generated_abc, list):
        generated_abc = "".join(generated_abc)
    cleaned = clean_abc(generated_abc, "./output/generated_abc_data.abc")
    print("Cleaning done.")
    print("Wrapping ABC body with header...")
    full_text = wrap_abc_body_with_header(cleaned , output_file)
    print("Wrapping done.")
    print("full_text:", full_text)
    print("Generated ABC saved to:", output_file)

if __name__ == "__main__":
    # Example usage+
    generate_abc("./dataset/input2.abc", "output2.abc", TOKENIZER_MODEL1200, MODEL_DIR1200, VOCAB_SIZE1200)