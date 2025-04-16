# abc_music_pipeline.py

import os
import torch
import youtokentome as yttm
from music21 import converter
from transformers import EncoderDecoderModel
import subprocess

# === Step 1: Convert MIDI to ABC ===
def midi_to_abc(midi_path):
    try:
        score = converter.parse(midi_path)
        abc_path = score.write('abc')
        with open(abc_path, 'r') as f:
            abc_content = f.read()
        return abc_content
    except Exception as e:
        print(f"[ERROR] MIDI to ABC failed: {e}")
        return None

# === Step 2: Tokenize ABC ===
tokenizer = yttm.BPE(model='abc_tokenizer.model')

def tokenize_abc(abc_text):
    return tokenizer.encode(abc_text, output_type=yttm.OutputType.ID)

# === Step 3: Generate ABC using Transformer ===
model = EncoderDecoderModel.from_pretrained('./abc_transformer_model')

def generate_continuation(input_ids, max_len=256):
    input_tensor = torch.tensor([input_ids])
    output = model.generate(
        input_tensor,
        max_length=max_len,
        num_beams=5,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.subword_to_id('<eos>')
    )
    return output[0].tolist()

# === Step 4: Decode ABC Tokens ===
def decode_tokens(token_ids):
    return tokenizer.decode(token_ids)

# === Step 5: Convert ABC to MIDI ===
def abc_to_midi(abc_text, output_path="generated.mid"):
    try:
        with open("temp.abc", "w") as f:
            f.write(abc_text)
        score = converter.parse("temp.abc")
        score.write('midi', fp=output_path)
        return output_path
    except Exception as e:
        print(f"[ERROR] ABC to MIDI failed: {e}")
        return None

# === Step 6: Convert MIDI to WAV ===
def midi_to_wav(midi_path, soundfont_path, output_wav):
    try:
        command = [
            'fluidsynth',
            '-ni',
            soundfont_path,
            midi_path,
            '-F',
            output_wav,
            '-r',
            '44100'
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"[ERROR] MIDI to WAV failed: {e}")

# === Final Pipeline ===
def full_pipeline(midi_file, sf2_path):
    abc = midi_to_abc(midi_file)
    if abc is None:
        return

    input_ids = tokenize_abc(abc)
    generated_ids = generate_continuation(input_ids)
    new_abc = decode_tokens(generated_ids)

    midi_out = abc_to_midi(new_abc, "generated.mid")
    if midi_out:
        midi_to_wav(midi_out, sf2_path, "output.wav")
        print(" Music generation complete: output.wav")

if __name__ == "__main__":
    full_pipeline("example.mid", "FluidR3_GM.sf2")

def wrap_abc_body_with_header(abc_body: str) -> str:
    header = (
        "X:1\n"
        "T:Generated Tune\n"
        "M:4/4\n"
        "L:1/8\n"
        "Q:1/4=120\n"
        "K:C\n"
    )
    return header + abc_body.strip()
