# Deep Learning Final Project

Setup:

- create and install environment:

venv:

```
python3.10 -m venv dl-env
source dl-env/bin/activate
```

conda:

```
conda create --name dl python=3.10
conda activate dl
```

- install dependencies:

```
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Usage:
