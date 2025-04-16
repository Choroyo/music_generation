import youtokentome as yttm

yttm.BPE.train(
    data='./dataset/all_abc_data.txt',
    model="abc_tokenizer1200.model",
    vocab_size=1200,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
print("Tokenizer trained and saved as abc_tokenizer.model vocab size: ", 1200)