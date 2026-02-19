from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Paths (match your folders)
# -------------------------
BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "veltri" / "all_veltri.csv"

OUT_DIR = BASE / "embeddings" / "veltri"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "veltri_embeddings.npy"
META_PATH = OUT_DIR / "veltri_metadata.csv"

# -------------------------
# Model (pretrained ProteoGPT)
# -------------------------
MODEL_ID = "nferruz/ProtGPT2"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

# ✅ Your actual column names
SEQ_COL = "aa_seq"   # peptide strings
LABEL_COL = "AMP"    # 1/0 label

sequences = df[SEQ_COL].astype(str).tolist()

# Convert AMP label to True/False
y = df[LABEL_COL]
if y.dtype == bool:
    is_amp = y.astype(bool)
elif pd.api.types.is_numeric_dtype(y):
    is_amp = y.astype(int).eq(1)
else:
    # fallback if it's strings like "AMP" / "non-AMP"
    is_amp = y.astype(str).str.strip().str.lower().isin(
        ["1", "true", "t", "amp", "antimicrobial", "yes", "y", "pos", "positive"]
    )

# -------------------------
# Load tokenizer + model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# ✅ Required fix for GPT-style tokenizers (no pad token by default)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

# -------------------------
# Mean pooling (ignore padding)
# -------------------------
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B,1)
    return summed / counts

# -------------------------
# Embed sequences
# -------------------------
batch_size = 16
vecs = []

with torch.no_grad():
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]

        tok = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        out = model(**tok)  # out.last_hidden_state: (B,T,H)
        pooled = mean_pool(out.last_hidden_state, tok["attention_mask"])  # (B,H)

        vecs.append(pooled.cpu().numpy().astype("float32"))
        print(f"Embedded {min(i+batch_size, len(sequences))}/{len(sequences)}")

X = np.vstack(vecs)

# -------------------------
# Save outputs
# -------------------------
np.save(EMB_PATH, X)
pd.DataFrame({
    "sequence": sequences,
    "is_amp": is_amp.astype(bool)
}).to_csv(META_PATH, index=False)

print("Saved embeddings:", EMB_PATH, X.shape)
print("Saved metadata:", META_PATH)
