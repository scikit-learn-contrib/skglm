## Installation

1. checkout branch
```shell
# add remote if it does't exist (check with: git remote -v)
git remote add Badr-MOUFAD https://github.com/Badr-MOUFAD/skglm.git

git fetch Badr-MOUFAD skglm-gpu

git checkout skglm-gpu
```

2. create then activate``conda`` environnement
```shell
# create
conda create -n another-skglm-gpu python=3.7

# activate env
conda activate another-skglm-gpu
```

3. install ``skglm`` in editable mode
```shell
pip install skglm -e .
```

4. install dependencies
```shell
# cupy
conda conda install -c conda-forge cupy cudatoolkit=11.5

# jax
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```
