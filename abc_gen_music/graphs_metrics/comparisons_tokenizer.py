import youtokentome as yttm
from collections import Counter
import matplotlib.pyplot as plt

# Load both tokenizers
tokenizer_small = yttm.BPE(model="./abc_tokenizer600.model")
tokenizer_large = yttm.BPE(model="./abc_tokenizer3000.model")

# Load ABC data
with open("./dataset/all_abc_data.txt", "r", encoding="utf-8") as f:
    abc_data = f.read()

# Encode into subword tokens
tokens_small = tokenizer_small.encode(abc_data, output_type=yttm.OutputType.SUBWORD)
tokens_large = tokenizer_large.encode(abc_data, output_type=yttm.OutputType.SUBWORD)

# Count token frequency
freq_small = Counter(tokens_small)
freq_large = Counter(tokens_large)

# Get top 30 tokens
top_small = freq_small.most_common(30)
top_large = freq_large.most_common(30)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

bar_width = 0.4
x = list(range(len(top_small)))

ax.bar([i - bar_width/2 for i in x], [t[1] for t in top_small], width=bar_width, label='Vocab 600')
ax.bar([i + bar_width/2 for i in x], [t[1] for t in top_large], width=bar_width, label='Vocab 3000')

ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in top_small], rotation=90)
ax.set_ylabel("Token Frequency")
ax.set_title("Top 30 Tokens: Vocab 600 vs Vocab 3000")
ax.legend()

plt.tight_layout()
plt.savefig("tokenizer_comparison.png")
print(" Plot saved as tokenizer_comparison.png")
