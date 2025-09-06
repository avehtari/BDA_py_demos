# Bayesian Data Analysis Python Demos

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/avehtari/bda_py_demos) to interactively run the IPython Notebooks in the browser.

This repository contains some Python demos for the book [Bayesian Data
Analysis, 3rd ed by Gelman, Carlin, Stern, Dunson, Vehtari, and Rubin (BDA3)](http://www.stat.columbia.edu/~gelman/book/). See also [Bayesian Data Analysis course material](https://github.com/avehtari/BDA_course_Aalto).

Currently there are demos for BDA3 Chapters 2, 3, 4, 5, 6, 10 and 11. Furthermore, [cmdstanpy](https://mc-stan.org/cmdstanpy/) is also demoed.

Demos are in jupyter notebook (.ipynb) format. These can be directly previewed in GitHub without need to install or run anything.

Corresponding demos were originally written for [Matlab/Octave](https://github.com/avehtari/BDA_m_demos) by [Aki Vehtari](http://users.aalto.fi/~ave/) and translated to Python by Tuomas Sivula. Some improvements were contributed by Pellervo Ruponen and Lassi Meronen. There are also corresponding [R demos](https://github.com/avehtari/BDA_R_demos).


## Requirements

- python
- ipython
- numpy
- scipy
- matplotlib 
- pandas (for some demos)
- cmdstanpy (for some demos)
- ArviZ (for some demos)


You can install all necessary packages with `environment.yml` file using [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/):

```bash
conda env create -f environment.yml
``` 

or 

```bash
mamba env create -f environment.yml
``` 