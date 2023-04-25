## Installation


1. create then activate ``conda`` environnement
```shell
# create
conda create -n skglm-jax python=3.7

# activate env
conda activate skglm-jax
```

2. install ``skglm`` in editable mode
```shell
pip install skglm -e .
```

3. install dependencies
```shell
# jax
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```
