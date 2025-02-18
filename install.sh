#!/bin/bash
pip3 install virtualenv
rm -rf .venv
python3 -m virtualenv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install --no-cache-dir -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install --no-cache-dir wandb hydra-core flax optax
deactivate
