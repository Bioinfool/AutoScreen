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

# Smoke library baked into the image; compose also mounts ./data for larger sets.
COPY data/Enamine10k.csv.gz /workspace/data/Enamine10k.csv.gz
COPY data/Enamine10k_scores.csv.gz /workspace/data/Enamine10k_scores.csv.gz
COPY data/Enamine10k_moo.csv.gz /workspace/data/Enamine10k_moo.csv.gz

ENV AUTOSCREEN_TRUTH_MOO=data/Enamine10k_moo.csv.gz

RUN pip install --no-cache-dir -e ".[robot,dev,prep]" \
 && python /workspace/scripts/prep_fps_serial.py --name Enamine10k

CMD ["bash"]
