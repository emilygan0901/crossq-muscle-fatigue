#!/usr/bin/env bash
# Rebuild the MyoSuite environment (env-dev) and the patched CrossQ code on a Kaggle GPU session.
# This is the code of "cell 1" of the Kaggle notebooks (see notes/run_notes.md).
#
#   bash environment/setup_kaggle.sh            # run from /kaggle/working, Internet on, GPU T4 x2
#
# Result: /kaggle/working/kit (this repository), /kaggle/working/CrossQ-dev (authors' code at
# commit a318a26 + patches/crossq_coursework.patch) and /kaggle/working/env-dev (Python 3.11.5 with
# the exact versions in environment/requirements-myosuite.txt).
set -e
cd /kaggle/working
pip install -q uv

if [ -d kit ]; then (cd kit && git pull -q); else git clone -q https://github.com/emilygan0901/crossq-muscle-fatigue.git kit; fi

[ -d CrossQ-dev ] || git clone -q https://github.com/adityab/CrossQ.git CrossQ-dev
cd CrossQ-dev
git checkout -q a318a266679bec3f9d93c44e029860a6da077c89
git checkout -q -- .
git apply ../kit/patches/crossq_coursework.patch
cd ..

if [ ! -x env-dev/bin/python ]; then
  uv venv -q --python 3.11.5 env-dev
  grep -v '^-e \|^#' kit/environment/requirements-myosuite.txt > req_dev.txt
  uv pip install -q --python env-dev/bin/python --no-deps -r req_dev.txt \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  uv pip install -q --python env-dev/bin/python --no-deps -e CrossQ-dev
  # torch 2.0.1 (pulled in by MyoSuite) ships CUDA 11 libraries that overwrite files JAX needs,
  # so the CUDA 12 solver and cuDNN are reinstalled last.
  uv pip install -q --python env-dev/bin/python --no-deps --reinstall \
      nvidia-cusolver-cu12==11.7.5.82 nvidia-cudnn-cu12==8.9.7.29
fi
env-dev/bin/python -c "import jax, mujoco, myosuite; print('JAX', jax.__version__, jax.devices()); print('MuJoCo', mujoco.__version__)"
