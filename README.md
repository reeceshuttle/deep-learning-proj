# Deep Learning Final Project

### Setup:

1. create and activate environment:

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

2. install dependencies:

```
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Usage:

We use the OLMo 7b and 1b models.
step: which revision to use.

example:

```
python main.py --size=7b --step=500000 {args}
```

python main.py --size=1b --step=500000 --reproduce_paper=True --quantization_method=awq
