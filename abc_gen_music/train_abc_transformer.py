#This code will train the transformer from abc data

import os
import youtokentome as yttm
from transformers import (
    BertConfig, BertModel, GPT2Config, GPT2LMHeadModel,
    EncoderDecoderModel, EncoderDecoderConfig, TrainingArguments,
    Trainer, TrainingArguments, TrainerCallback
)
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
import torch.multiprocessing as mp
from parameter import (TOKENIZER_MODEL3000, TOKENIZER_MODEL1200, MODEL_DIR3000,
                       MODEL_DIR1200, MODEL_DIR600,TOKENIZER_MODEL600, DATASET1200, 
                       DATASET3000, DATASET600, VOCAB_SIZE3000, VOCAB_SIZE1200, 
                       VOCAB_SIZE600)


# === Callback personalizado para imprimir progreso y motivación ===
class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 1000 == 0:
            print(f" STEP {state.global_step}] ¡This is training!")

def collate_fn(batch,device):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

# === Model Definition ===
def build_encoder_decoder_model(vocab_size, mode="bert-gpt2"):
    if mode == "bert-gpt2":
        encoder_config = BertConfig(vocab_size=vocab_size)
        decoder_config = GPT2Config(vocab_size=vocab_size, is_decoder=True, add_cross_attention=True)

        encoder = BertModel(encoder_config)
        decoder = GPT2LMHeadModel(decoder_config)

        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    elif mode == "bert-bert":
        encoder_config = BertConfig(vocab_size=vocab_size)
        decoder_config = BertConfig(vocab_size=vocab_size, is_decoder=True, add_cross_attention=True)

        model_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        model = EncoderDecoderModel(config=model_config)

    elif mode == "gpt2-gpt2":
        encoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=6, n_head=8, n_embd=512,  # You can customize
            add_cross_attention=False  # Encoder doesn't use this
        )

        decoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=6, n_head=8, n_embd=512,
            is_decoder=True,
            add_cross_attention=True  # Decoder needs this to attend to encoder
        )
        decoder = GPT2LMHeadModel(decoder_config)
        # Build encoder-decoder model
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    else:
        raise ValueError("Unknown model type")

    return model

# === Main training ===
def train(tokenizer, vocab_size, model_dir, dataset_dir, models="bert-gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Usando dispositivo: {device.upper()}")

    # Load tokenizer
    tokenizer = yttm.BPE(model=tokenizer)

    # Load dataset
    dataset = torch.load(dataset_dir)  #
    print(f"[INFO] Cargados {len(dataset)} ejemplos musicales (ABC samples)")

    # Build model
    model = build_encoder_decoder_model(vocab_size=vocab_size, mode=models)
    model.to(device)  # <-- Esto garantiza que el modelo se mueva a GPU si está disponible

    # Resize decoder embeddings BEFORE training
    model.decoder.resize_token_embeddings(vocab_size)
    model.config.decoder_start_token_id = tokenizer.subword_to_id('<s>')
    model.config.pad_token_id = tokenizer.subword_to_id('<pad>')
    model.config.eos_token_id = tokenizer.subword_to_id('</s>')

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),  # Usa FP16 si hay GPU
        logging_steps=300,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,        # 👈 Esto ya está bien
        disable_tqdm=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=None,
        data_collator=lambda batch: collate_fn(batch, device),
        callbacks=[CustomCallback()]  # Agregamos el callback aquí
    )

    print("[INFO] Starting training...")
    trainer.train(resume_from_checkpoint=True)  # <-- Esto reanuda el entrenamiento desde el último checkpoint
    print(f"[DONE] Model saved to {model_dir}")
    model.encoder.save_pretrained(os.path.join(model_dir, "encoder"))
    model.decoder.save_pretrained(os.path.join(model_dir, "decoder"))


if __name__ == "__main__":
    train(TOKENIZER_MODEL1200, VOCAB_SIZE1200, MODEL_DIR1200, DATASET1200, models ="bert-gpt2")
    train(TOKENIZER_MODEL3000, VOCAB_SIZE3000, MODEL_DIR3000, DATASET3000, models ="bert-gpt2")

