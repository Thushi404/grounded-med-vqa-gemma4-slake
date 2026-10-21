# grounded-med-vqa-gemma4-slake

# Grounded Medical Visual Question Answering with Gemma 4 E2B

Fine-tuning and evaluation of Gemma 4 E2B on the English SLAKE visual question-answering data. The model is trained to answer radiology questions and return an optional normalized image-region box. KG questions use relation-aware retrieval from SLAKE's English medical knowledge graph.

## Project artifacts

- [Final experiment report](FINAL_REPORT.md)
- Prepared dataset: [Hugging Face dataset repository](https://huggingface.co/datasets/Thushi2003/slake-grounded-vqa-gemma4-e2b)
- LoRA adapter: [Hugging Face model repository](https://huggingface.co/Thushi2003/slake-grounded-vqa-gemma4-e2b-lora)

## Dataset splits

| Split | English questions | Positive-region boxes |
|---|---:|---:|
| Train | 4,919 | 370 |
| Validation | 1,053 | 69 |
| Test | 1,061 | 78 |

## Main findings

- On a seeded 100-question validation sample, fine-tuning reached 74% exact answer match versus 29% zero-shot.
- On the full test split without retrieval, fine-tuned answer match was 741/1,061 (69.8%; 95% Wilson CI 67.0%–72.5%).
- Localization at IoU ≥ 0.5 was 0/78 on the test grounded questions.
- On a relation-balanced sample of 49 test KG questions, exact answer match was 26/49 without retrieval and 22/49 with top-five retrieved KG facts.

The current retriever can return facts for the wrong image-specific organ or disease. The report describes the evaluation samples, confidence intervals, retrieval limitations, and failure examples.

## Training configuration

Base model: `google/gemma-4-E2B-it`. Fine-tuning used 4-bit quantization and rank-8 LoRA on language-model attention projections. The final run used one epoch and 1,230 optimizer steps.

## Notes

This repository summary describes the completed experiments; include the actual training/evaluation notebook and preprocessing source from the working VS Code project when preparing the code repository. Results are experimental and are not intended for clinical decision-making.

