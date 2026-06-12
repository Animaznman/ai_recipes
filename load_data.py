from datasets import load_dataset
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

PG_USER     = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST     = os.getenv("PG_HOST")
PG_PORT     = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE")

missing = [k for k, v in {
    "PG_USER": PG_USER, "PG_PASSWORD": PG_PASSWORD,
    "PG_HOST": PG_HOST, "PG_DATABASE": PG_DATABASE
}.items() if not v]
if missing:
    raise ValueError(f"Missing required .env variables: {missing}")

engine = create_engine(
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    "?sslmode=require"
)

# Load from HuggingFace
ds = load_dataset("GPTasty/PAID-recipes-normalized")
df = ds["train"].to_pandas()

print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Drop all embedding columns (numpy float32 arrays)
embedding_cols = [
    col for col in df.columns
    if df[col].dtype == object
    and not df[col].dropna().empty
    and isinstance(df[col].dropna().iloc[0], np.ndarray)
    and df[col].dropna().iloc[0].dtype.kind == 'f'  # float arrays = embeddings
]
print(f"Dropping {len(embedding_cols)} embedding columns: {embedding_cols}\n")
df = df.drop(columns=embedding_cols)

# Convert remaining numpy string/object arrays to Python lists
for col in df.columns:
    if df[col].dtype == object:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(sample, np.ndarray):
            print(f"  Converting array column to list: {col}")
            df[col] = df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )

print(f"\nClean shape after dropping embeddings: {df.shape}")
print(f"Loading {len(df)} rows into Postgres...")

df.to_sql("recipes", engine, if_exists="replace", index=False)
print(f"Done! Loaded {len(df)} rows successfully.")