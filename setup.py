import os
from setuptools import setup, find_packages


version = None
with open(os.path.join('skglm', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'skglm'
DESCRIPTION = 'A fast and modular scikit-learn replacement for generalized linear models'
with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Mathurin Massias'
MAINTAINER_EMAIL = 'mathurin.massias@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/skglm.git'
VERSION = version
URL = 'https://contrib.scikit-learn.org/skglm'

setup(name=DISTNAME,
      version=version,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=find_packages(),
      install_requires=['numpy>=1.12', 'numba',
                        'scipy>=0.18.0', 'scikit-learn>=1.0']
      )
