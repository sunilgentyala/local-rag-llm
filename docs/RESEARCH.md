# Research: RAG vs. Fine-Tuning — Strategy, Trade-offs, and Future Roadmap

---

## 1. The Core Question

When customizing a large language model for a specific knowledge domain, two primary strategies exist:

| Strategy | Description |
|---|---|
| **Retrieval-Augmented Generation (RAG)** | Keep the base model frozen. At query time, retrieve relevant content from an external knowledge store and inject it into the prompt as context. |
| **Fine-Tuning** | Update the model's weights using a domain-specific dataset so the model internalizes the knowledge directly. |

This document explains why RAG was chosen for this project and maps the conditions under which fine-tuning becomes the right next step.

---

## 2. Why RAG for This Setup

### 2.1 Hardware Constraints

This application is designed to run on a **personal laptop** without a dedicated GPU. The implications are significant:

| Requirement | RAG | Full Fine-Tuning |
|---|---|---|
| GPU required? | No — CPU inference is viable | Yes — minimum 1x NVIDIA T4 (16 GB VRAM) |
| Training time on laptop CPU | N/A | Days to weeks — impractical |
| Disk storage for model weights | ~5 GB (quantized llama3) | 5 GB base + adapter weights |
| Working memory (RAM) | 8–16 GB sufficient | 16–80 GB depending on method |

A quantized `llama3` (Q4_K_M format, ~4.7 GB) runs at 5–15 tokens/sec on a modern laptop CPU — slow but usable. Fine-tuning even with QLoRA requires a GPU.

### 2.2 Data Availability

Fine-tuning requires a **curated, labeled dataset** in a specific format (typically instruction-response pairs). RAG requires only raw documents. For most personal use cases:

- Documents are readily available (PDFs, notes, reports)
- Labeled instruction-response pairs must be manually created — labor-intensive
- RAG allows immediate deployment; fine-tuning requires a dataset pipeline first

### 2.3 Knowledge Updateability

| Scenario | RAG | Fine-Tuning |
|---|---|---|
| New document added | Upload PDF → re-embed (minutes) | Requires full retraining cycle |
| Document removed | Delete from ChromaDB (seconds) | Impossible without retraining |
| Knowledge cutoff | No cutoff — real-time updates | Frozen at training time |

RAG's dynamic knowledge base is a decisive advantage for document-centric workflows.

### 2.4 Privacy

Both approaches can be run locally. However, fine-tuning via cloud services (Google Colab, AWS SageMaker) requires uploading training data to third-party infrastructure. RAG with Ollama keeps all data on-device at all times.

### 2.5 Cost

| Approach | Estimated Cost |
|---|---|
| RAG (local, this project) | **$0.00** — runs on existing hardware |
| OpenAI GPT-4 API (per query) | ~$0.01–$0.10 per query |
| Google Colab Pro (fine-tuning) | ~$9.99/month |
| AWS SageMaker fine-tuning | $1.00–$15.00+/hour |

---

## 3. When Fine-Tuning Becomes the Right Choice

RAG has meaningful limitations. Fine-tuning becomes appropriate when:

1. **Consistent stylistic output is required** — e.g., the model must always respond in a specific format, persona, or tone that retrieval cannot enforce reliably.
2. **Reasoning patterns need to change** — not just knowledge, but how the model reasons about a domain (e.g., medical diagnostic logic, legal interpretation style).
3. **Latency is critical** — fine-tuning eliminates the retrieval step, reducing end-to-end latency.
4. **Large labeled datasets exist** — when thousands of verified instruction-response pairs are available, fine-tuning captures patterns that prompting cannot.
5. **Context window constraints** — if Top-K retrieval still misses critical information, fine-tuning the model on that information removes the dependency.

---

## 4. Future Roadmap: QLoRA Fine-Tuning on Google Colab

When the conditions above are met, the recommended upgrade path for this project is **QLoRA fine-tuning** using [Unsloth](https://github.com/unslothai/unsloth) on Google Colab.

### 4.1 What is QLoRA?

**QLoRA** (Quantized Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) technique that:

- Loads the base model in **4-bit quantization** (drastically reducing VRAM usage)
- Adds small trainable **LoRA adapter matrices** to the attention layers
- Only the adapter weights (~1–10% of total parameters) are updated during training
- After training, adapters can be merged back into the base model or used separately

| Property | Full Fine-Tuning | QLoRA |
|---|---|---|
| VRAM required (7B model) | ~56 GB | ~6–10 GB |
| Trainable parameters | 100% | 1–3% |
| Training speed | Slow | 2–5x faster with Unsloth |
| Quality vs. full fine-tune | — | ~97% of full fine-tune quality |

### 4.2 Why Unsloth?

[Unsloth](https://github.com/unslothai/unsloth) is an open-source library that makes QLoRA fine-tuning **2–5x faster** and uses **60% less VRAM** than standard HuggingFace + PEFT workflows. It supports llama3, Mistral, Gemma, and Phi models natively.

### 4.3 Google Colab Setup (Future Notebook Outline)

```python
# Step 1 — Install Unsloth (Colab runtime: GPU T4 or A100)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

# Step 2 — Load base model in 4-bit
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name  = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit   = True,
)

# Step 3 — Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,          # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
)

# Step 4 — Prepare dataset (Alpaca format)
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_domain_data.jsonl", split="train")

# Step 5 — Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset= dataset,
    dataset_text_field = "text",
    max_seq_length     = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps   = 5,
        max_steps      = 200,
        learning_rate  = 2e-4,
        fp16           = True,
        output_dir     = "outputs",
    ),
)
trainer.train()

# Step 6 — Save and export to Ollama-compatible GGUF
model.save_pretrained_gguf("llama3-finetuned", tokenizer, quantization_method="q4_k_m")
```

### 4.4 Dataset Format (Alpaca Instruction Format)

```json
{
  "instruction": "Summarize the following policy document section.",
  "input": "Section 4.2: All employees must complete annual compliance training...",
  "output": "Employees are required to complete compliance training every year."
}
```

### 4.5 Recommended Colab Runtime

| Use Case | Colab Tier | GPU | Est. Training Time (200 steps) |
|---|---|---|---|
| Experimentation | Free | T4 (16 GB) | ~15–25 min |
| Full dataset run | Colab Pro | A100 (40 GB) | ~5–10 min |

---

## 5. Decision Framework Summary

```
Start here
    │
    ▼
Do you have labeled instruction-response pairs?
    │
    ├─ NO ──► Use RAG (this project) ◄── Start here
    │
    └─ YES
         │
         ▼
    Do you have GPU access (Colab / cloud)?
         │
         ├─ NO ──► Use RAG + prompt engineering
         │
         └─ YES
              │
              ▼
         Is the task style/reasoning or knowledge?
              │
              ├─ Knowledge ──► RAG is still faster and cheaper
              │
              └─ Style/Reasoning ──► QLoRA fine-tuning with Unsloth
```

---

## 6. References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [Ollama Model Library](https://ollama.com/library)
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
