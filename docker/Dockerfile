FROM python:3.11-slim

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

RUN pip install --no-cache-dir \
    numpy scipy scikit-learn h5py pymoo PyYAML tqdm rdkit httpx fastapi uvicorn pytest

COPY autoscreen /workspace/autoscreen
COPY robot_mock /workspace/robot_mock
COPY scripts /workspace/scripts
COPY configs /workspace/configs
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md

COPY molpal/libraries/Enamine10k.csv.gz /workspace/molpal/libraries/Enamine10k.csv.gz
COPY molpal/data/Enamine10k_scores.csv.gz /workspace/molpal/data/Enamine10k_scores.csv.gz
COPY molpal/data/Enamine10k_moo.csv.gz /workspace/molpal/data/Enamine10k_moo.csv.gz

RUN pip install --no-cache-dir -e ".[robot,dev]" \
 && python /workspace/scripts/prep_fps_serial.py

CMD ["bash"]
