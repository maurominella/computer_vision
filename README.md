
# Rendering Defects Prototype — Deliverables

This package includes three notebooks, a sample dataset structure, and an overlay script
to demonstrate the end-to-end pipeline proposed to GUESS.

## Files
- 01_region_proposals_vision.ipynb: Dense Captions & Brands to produce region proposals.
- 02_llm_triage_responses.ipynb: Multimodal triage with Azure OpenAI Responses API.
- 03_automl_object_detection.ipynb: Submit AutoML job for object detection baseline.
- samples/train.jsonl, samples/val.jsonl: Sample JSONL labels (placeholders).
- mltable/train/MLTable, mltable/val/MLTable: MLTable definitions referencing JSONL.
- overlay_bboxes.py: Draw detector/LLM bboxes+notes over the image.

## Quick Start (Conceptual)
1. Run notebook 01 to generate artifacts/proposals.json for an image.
2. Run notebook 02 to obtain artifacts/llm_triage.json (structured suspects).
3. Prepare a small labeled set, update MLTable, and run notebook 03 to train a baseline.
4. Use overlay_bboxes.py to visualize defect boxes on the image.

> Replace placeholder endpoints/keys/URLs with your actual Azure resources before running.


## UV Installation
- On Linux / MAC --> `curl -LsSf https://astral.sh/uv/install.sh | sh`
- On Windows --> `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Environment preparation
- *CD* into the folder
- Create the environment: `uv init . --python 3.13`.
- Add libraries (bad method): `uv add azure-ai-vision-imageanalysis openai azure-ai-ml==1.31.0 azure-identity python-dotenv matplotlib opencv-python numpy requests Pillow jupyter`.
- Add libraries (better method): `uv add $(cat requirements.txt)`.
- Syncrhonize to create the file structure: `uv sync`.
- Activate the environment:
  - on Linux/MC --> `source .venv/bin/activate`.
  - on Windows --> `.venv\Scripts\activate.ps1`.
- To deactivate --> `deactivate`.
- To create the kernel for the jupyter notebooks: `python -m ipykernel install --name poc_guess --use`
