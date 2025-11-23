# Installation Instruction

You can create an anaconda environment called `dbgroup ` as below. For linux, you need to install `libopenexr-dev` before creating the environment.

```bash
sudo apt-get install libopenexr-dev # for linux
conda create -n dbgroup python=3.8
conda activate dbgroup
conda install openblas-devel -c anaconda
```

Step 1: install PyTorch(cuda 10.2 / cuda 11.1):

```bash
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Step 2: install all the remaining dependencies:

```bash
pip install -r requirements.txt
```
