# Adapted to work with InferenceData from ArviZ
# Original version from
# Copyright: Michael Betancourt <https://betanalpha.github.io/writing/>
# License: BSD (3 clause)
# See also http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html

import arviz as az

def check_div(idata):
    """Check transitions that ended with a divergence"""
    divergent = idata.sample_stats.diverging.values
    n = divergent.sum()
    N = divergent.size
    if n > 0:
        print(f'{n} of {N} iterations ended with a divergence ({100 * n / N:.1f})')
        print('Try running with larger adapt_delta to remove the divergences')

def check_treedepth(idata, max_depth = 10):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    depths = idata.sample_stats.tree_depth.values
    n = (depths == max_depth).sum()
    N = depths.size
    if n > 0:
        print((f'{n} of {N} iterations saturated the maximum tree depth of {max_depth}'
            + f' ({100 * n / N:.1f}%)'))
        print('Run again with max_depth set to a larger value to avoid saturation')

def check_energy(idata):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)"""
    bfmi = az.bfmi(idata)
    for chain_num, b in enumerate(bfmi):
        if b < 0.2:
            print('Chain {}: E-BFMI = {}'.format(chain_num, b))
            print('E-BFMI below 0.3 indicates you may need to reparameterize your model')
