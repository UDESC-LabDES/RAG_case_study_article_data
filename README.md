# RAG Pipeline for Tomato-Crop Technical Questions

> **Project status:** research prototype / work in progress. The notebook contains experiment-specific paths and some cells that still need to be aligned before the complete pipeline can be run from a clean environment.

## Overview

This project evaluates a retrieval-augmented generation (RAG) pipeline for questions about tomato cultivation and plant health. It uses prepared technical-text files, retrieves relevant passages, generates answers with language models, and compares those answers with reference answers using BERTScore.

The current notebook experiments with two document collections:

- **Without descriptions:** `textos_semdescricao_tomate`
- **With descriptions:** `textos_comdescricao_tomate`

The notebook does not scrape the source website itself. It expects preprocessed `.txt` documents to be available in Google Drive. The project context identifies EpagriTEC/Sedimob as the intended source of the technical material: [EpagriTEC consultation page](https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=epagriTec).

At a high level, the pipeline:

1. Loads the prepared text documents and a question/reference-answer dataset.
2. Splits each document into overlapping chunks and adds title/URL metadata when available.
3. Embeds the chunks with BGE-M3 and creates FAISS indexes.
4. Retrieves candidate contexts and uses a BGE reranker when the initial FAISS distance does not meet the configured threshold.
5. Generates answers with an Ollama-hosted model (currently Ministral-8B is enabled in the generation configuration; other configurations are commented out) or optionally with Gemini if the configuration is changed.
6. Scores generated answers against `ground_truth` with BERTScore, using XLM-RoBERTa and a local BERTimbau model.
7. Runs statistical summaries and comparison plots in later notebook cells.

## Execution environment

The supported execution path in the current code is **Google Colab with Google Drive mounted and a GPU runtime**. The notebook uses Colab-specific imports, shell commands such as `!pip`/`!apt-get`, and absolute paths under `/content/drive/MyDrive/LLM-TCC/`.

Use `full_code_english.ipynb` in Colab. The exported `full_code_english.py` is provided for inspection, but it is **not a standalone Python script** because it contains notebook magics and expects the Colab environment.

The retrieval, reranking, and BERTScore cells explicitly request `device="cuda"`; select a GPU runtime before running those stages. Model downloads can be several gigabytes, and the notebook also downloads and runs local language models through Ollama.

## Required input assets and paths

The current notebook hard-codes this base directory:

```text
/content/drive/MyDrive/LLM-TCC/
```

Create/use that directory in the Google Drive account connected to Colab, and place the required inputs there. The relevant expected assets are listed below; this is a description of the current code's path conventions, not a finalized repository organization plan.

| Asset | Expected location | Purpose |
|---|---|---|
| Golden question/reference dataset | `golden_dataset_tomate.csv` | Provides the evaluation questions and reference answers. The code expects at least `question` and `ground_truth` columns. |
| Text collection without descriptions | `textos_semdescricao_tomate/*.txt` | Source documents for the no-description retrieval condition. |
| Text collection with descriptions | `textos_comdescricao_tomate/*.txt` | Source documents for the with-description retrieval condition. |
| Local BERTimbau model | `modelos_locais/bert-base-portuguese-cased/` | Local model directory for the Portuguese BERTScore run; it must contain `config.json` and the model/tokenizer files. |
| Hugging Face cache | `huggingface_cache/` | Persistent cache configured for Hugging Face Transformers, Sentence-Transformers, datasets, and PyTorch downloads. |

Each source `.txt` file is expected to contain its document text. The chunking function looks for `Título:` and `URL:` in the first five lines, removes the detected header lines from the chunked body, and then prefixes the title and URL to every chunk. The filename stem is used as the `cdEstrutura` identifier.

When the processed-document caches do not exist, the notebook attempts to create these files in the base directory:

- `textos_semdescricao_tomate.pkl`
- `textos_comdescricao_tomate.pkl`

The FAISS indexes are created under names derived from the base path, including:

- `dbs_faiss_textos_semdescricao_tomate/`
- `dbs_faiss_textos_comdescricao_tomate/`

These are generated assets, not the original text collection.

## Models and where they come from

### Retrieval embedding model: BGE-M3

- **Model ID:** [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
- **Use:** Converts the question and document chunks into embeddings for FAISS retrieval.
- **Download behavior:** The notebook asks Sentence-Transformers to load this model by its Hugging Face model ID. It is downloaded automatically on first use and cached under `huggingface_cache/` according to the environment variables configured by the notebook.

### Retrieval reranker: BGE Reranker Large

- **Model ID:** [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large)
- **Use:** Re-scores candidate question/context pairs when the initial FAISS distance is above the configured threshold.
- **Download behavior:** `CrossEncoder("BAAI/bge-reranker-large", device="cuda")` fetches the model from Hugging Face on first use and caches it. The current code sets the reranker score threshold to `0.7`, takes up to 20 candidates for reranking, and keeps up to 5 final contexts.

### BERTScore evaluation model 1: XLM-RoBERTa Large

- **Model ID in the notebook:** `xlm-roberta-large`
- **Model card:** [FacebookAI/xlm-roberta-large on Hugging Face](https://huggingface.co/FacebookAI/xlm-roberta-large)
- **Scoring library:** [`bert-score`](https://github.com/Tiiiger/bert_score), pinned to `0.3.13` in the installation cell.
- **Use:** Calculates precision, recall, and F1 for each generated answer against the corresponding `ground_truth` answer.
- **Download behavior:** The notebook passes the model ID to `BERTScorer`, so Transformers downloads the model files on first use. This evaluator is not stored in the project as a manually maintained local folder; it is downloaded from the Hugging Face Hub and cached.
- **Configuration:** `lang="pt"`, `rescale_with_baseline=False`, and a tokenizer/model length cap of 512 tokens are configured in the notebook.

### BERTScore evaluation model 2: BERTimbau Base

- **Expected local path:** `modelos_locais/bert-base-portuguese-cased/`
- **Official model card:** [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- **Use:** Provides the Portuguese-language model used in the second BERTScore run.
- **Important:** Unlike XLM-RoBERTa, the notebook **does not download BERTimbau in the evaluation cell**. It checks that `config.json` exists at the expected local path and raises an error if it does not. The intended model source is the Hugging Face repository above; the model repository needs to be downloaded/copied into the expected Google Drive directory first.

For example, in a Colab cell, the Hugging Face Hub client can download the model to the required location:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="neuralmind/bert-base-portuguese-cased",
    local_dir="/content/drive/MyDrive/LLM-TCC/modelos_locais/bert-base-portuguese-cased",
)
```

Do not commit model weights to Git unless you have deliberately chosen to version them and checked the repository size and model licensing terms. Keeping the model in Drive and documenting its source is the current approach assumed by the notebook.

### Answer-generation models

The setup cell installs Ollama and downloads the following models:

| Model | How it is fetched | Notes |
|---|---|---|
| Llama 3 | `ollama pull llama3` | Downloaded through the Ollama model registry. |
| Gemma 7B | `ollama pull gemma:7b` | Downloaded through the Ollama model registry. |
| Ministral-8B | Hugging Face repository [`bartowski/Ministral-8B-Instruct-2410-GGUF`](https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF), file `Ministral-8B-Instruct-2410-Q4_K_M.gguf` | The downloaded GGUF file is converted into an Ollama model named `ministral-8b`. |
| Gemini | Google Generative AI API | Optional; requires an API key and is used only when the selected model name contains `gemini`. It is not a local model. |

The active `CONFIGURACOES_PARA_RODAR` entries in the current notebook enable **Ministral-8B with and without document descriptions**, with `NUM_EXECUCOES = 30`. The Llama 3 and Gemma configurations are present but commented out. Adjusting those settings changes which generation experiments are run.

## Google Colab setup

1. Put the notebook in GitHub and open it in Colab, or upload `full_code_english.ipynb` directly to Colab.
2. In Google Drive, create/use `MyDrive/LLM-TCC/` and put the input files and directories described above in their expected locations.
3. In Colab, select a GPU runtime.
4. Open Colab **Secrets** and add `HF_TOKEN`. The notebook uses this token for Hugging Face login/downloads. Public model repositories may not require authentication, but the setup attempts to log in with the secret.
5. Add `GEMINI_API_KEY` only if you intend to run a Gemini configuration. The currently enabled Ministral configurations do not require the Gemini API key.
6. Run the notebook cells in order. The installation cell pins package versions; when it asks for a runtime restart, restart the runtime and then execute the setup cells again from the beginning.
7. Allow the Ollama setup cell to install Ollama, start the server, and download the configured generation models. The first load of embedding, reranking, and scoring models also downloads their weights.

Never paste API keys or Hugging Face tokens into a public notebook or commit them to Git. Use Colab Secrets.

## Notebook stages

Run the notebook sequentially because later cells depend on variables and files created earlier.

| Notebook cell | Purpose |
|---|---|
| 0 | Installs pinned Python package versions. The runtime must be restarted afterward if requested. |
| 1 | Mounts Google Drive, configures persistent Hugging Face/PyTorch caches, and loads Colab Secrets. |
| 2 | Installs Ollama and downloads the local answer-generation models. |
| 3 | Defines the embedding, FAISS, and LLM-loading helper functions. |
| 4 | Reads the `.txt` documents, creates/loads chunk caches, and creates FAISS indexes for both description conditions. |
| 5 | Retrieves contexts for the questions in `golden_dataset_tomate.csv`, applies the L2 threshold and reranker fallback, and saves the context CSVs. |
| 6 | Optional memory cleanup. |
| 7 | Generates answer CSVs for each enabled LLM/context configuration and run. |
| 8 | Calculates BERTScore using XLM-RoBERTa Large. |
| 9 | Calculates BERTScore using the local BERTimbau model. |
| 10–15 | Performs descriptive/statistical analysis and produces comparison plots. Check the notes below before relying on these cells for final results. |

## Generated outputs

The current code refers to or creates the following outputs under `MyDrive/LLM-TCC/`:

- `contextos_recuperados_com_decisao_L2_fallback_sem_descricao.csv`
- `contextos_recuperados_com_decisao_L2_fallback_com_descricao.csv`
- `respostas_geradas_pre_recuperadas_modelos_recentes/` — generated answer CSVs, one file per configuration/run
- `resultados_completos_bertscore_xlm_roberta4.csv` — consolidated XLM-RoBERTa BERTScore results
- `resultados_completos_bertscore4.csv` — filename currently written by the local BERTimbau evaluator
- `resultados_estatisticas_bertscore_por_pergunta_bertimbau4.csv` — per-question summary output referenced in the analysis code

The notebook can also save the critical-distance plots `CD_Diagram_BERTIMBAU.png` and `CD_Diagram_XLM-Roberta.png` in the current runtime working directory, depending on the statistical-analysis cells that are executed.

## Known issues to resolve before treating a run as reproducible

The notebook is still being developed. In particular:

1. **Answer directory mismatch:** the answer-generation cell writes to `respostas_geradas_pre_recuperadas_modelos_recentes/`, while both BERTScore cells currently read from `respostas_geradas_pre_recuperadas4/`. Make these paths consistent before running the evaluators, or they may find no generated answer files.
2. **BERTimbau result filename mismatch:** the BERTimbau evaluation cell writes `resultados_completos_bertscore4.csv`, but a later comparison cell tries to read `resultados_completos_bertscore_modelos4.csv`. Choose one filename and update the writer and all readers consistently.
3. **Chunking-cell variable typo in the English export:** the print statement uses `len(files_txt)`, although the list created immediately above is named `arquivos_txt`. If this line is reached when creating chunks, change it to `len(arquivos_txt)` before execution. The original Portuguese source file uses the correct variable name.
4. **Hard-coded statistical plot data:** the final critical-distance comparison cell defines example score arrays directly in the code. Those plots should not be presented as results calculated from the current CSV outputs unless the arrays are replaced with the actual experimental data.
5. **Path assumptions:** the notebook uses hard-coded Google Drive paths. If the folder name or location changes, update `base_path` and related paths in all dependent cells.

These issues are recorded here so readers can distinguish the intended workflow from details that remain to be fixed in the implementation.

## Reproducibility and interpretation notes

- The golden dataset is loaded from a CSV. The current notebook does not document how its questions and reference answers were selected or authored; that provenance should be added when finalized.
- The notebook's scraping/preprocessing stage is outside this code. The `.txt` files are assumed to have been created beforehand.
- The BERTScore numbers are based on the selected evaluation model and configuration. Scores from different evaluator models should be compared cautiously; XLM-RoBERTa and BERTimbau are different encoders.
- Do not interpret BERTScore alone as proof of factual correctness. It measures semantic similarity between generated answers and reference answers and should be read alongside domain-specific validation.
- The notebook contains research-specific settings and should be reviewed before use on a new dataset or a different GPU/runtime.

## Third-party model references

- BGE-M3: https://huggingface.co/BAAI/bge-m3
- BGE Reranker Large: https://huggingface.co/BAAI/bge-reranker-large
- XLM-RoBERTa Large: https://huggingface.co/FacebookAI/xlm-roberta-large
- BERTimbau Base: https://huggingface.co/neuralmind/bert-base-portuguese-cased
- Ministral GGUF repository used by the notebook: https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF
- BERTScore implementation: https://github.com/Tiiiger/bert_score
- Ollama: https://ollama.com/

Review the license terms on each model card and the licenses of the software dependencies before redistributing models or generated artifacts. A project-level license for this repository has not been specified in the notebook.
