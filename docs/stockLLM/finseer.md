# FinSeer Model Card

## Model details

**Model type:**

This is our first dedicated retriever for financial time-series forecasting, Financial TimeSeries Retriever (FinSeer).

**Paper or resources for more information:**

https://arxiv.org/pdf/2502.05878


## Intended use
**Primary intended uses:**
The primary use of FinSeer is research on financial time-series forecasting using retrieval-augmented generation (RAG) framework.

# How to use

## Installation

Install Package


    pip install InstructorEmbedding
    pip install -U FlagEmbedding
    pip install sentence-transformers==2.2.2
    pip install protobuf==3.20.0
    pip install yahoo-finance
    python -m pip install -U angle-emb
    pip install transformers==4.33.2  # UAE


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="TheFinAI/FinSeer")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("TheFinAI/FinSeer")
model = AutoModel.from_pretrained("TheFinAI/FinSeer")