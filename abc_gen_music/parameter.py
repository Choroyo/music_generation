import youtokentome as yttm
import os


# === Parameters ===
DATA_FILE = "./dataset/all_abc_data.txt"  # Your ABC data with full compositions
TOKENIZER_MODEL600 = "./tokenizers/abc_tokenizer600.model"
TOKENIZER_MODEL3000 = "./tokenizers/abc_tokenizer3000.model"
TOKENIZER_MODEL1200 = "./tokenizers/abc_tokenizer1200.model"  # Your ABC data with full compositions
MODEL_DIR600 = "./transformers/abc_transformer_model600"  # Directory to save the model
MODEL_DIR3000 = "./transformers/abc_transformer_model3000"
MODEL_DIR1200 = "./transformers/abc_transformer_model1200"
DATASET600 = "./dataset/dataset_tokenized/dataset600.pt"
DATASET1200 = "./dataset/dataset_tokenized/dataset1200.pt"
DATASET3000 = "./dataset/dataset_tokenized/dataset3000.pt"   # Directory to save the model
BLOCK_SIZE = 256
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}
VOCAB_SIZE3000 = 3000
VOCAB_SIZE600 = 600
VOCAB_SIZE1200 = 1200