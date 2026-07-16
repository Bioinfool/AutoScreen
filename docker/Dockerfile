# Lean, CPU-only image for the AutoScreen pipeline.
# AutoScreen has no MolPAL/TensorFlow/Ray runtime dependency; it only needs a
# small scientific-Python stack, so the image stays small and builds reliably.
FROM python:3.11-slim

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    h5py \
    pymoo \
    rdkit \
    tqdm

# Project code
COPY autoscreen /workspace/autoscreen
COPY scripts /workspace/scripts

# Example data (library SMILES, docking scores, multi-objective labels)
COPY molpal/libraries/Enamine10k.csv.gz /workspace/molpal/libraries/Enamine10k.csv.gz
COPY molpal/data/Enamine10k_scores.csv.gz /workspace/molpal/data/Enamine10k_scores.csv.gz
COPY molpal/data/Enamine10k_moo.csv.gz /workspace/molpal/data/Enamine10k_moo.csv.gz

# Precompute fingerprints so the image is plug-and-play out of the box
RUN python /workspace/scripts/prep_fps_serial.py

CMD ["bash"]
