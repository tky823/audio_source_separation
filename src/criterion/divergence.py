import numpy as np

EPS=1e-12

def kl_divergence(input, target, eps=EPS):
    """
    Args:
        input (C, ...)
    Returns:
        loss (...)
    """
    _input, _target = input.copy(), target.copy()
    _input[_input < eps] = eps
    _target[_target < eps] = eps

    ratio = _target / _input
    loss = _target * np.log(ratio)
    loss = loss.sum(dim=0)

    return loss

def is_divergence(input, target, eps=EPS):
    """
    Args:
        input (...)
    """
    _input, _target = input.copy(), target.copy()
    _input[_input < eps] = eps
    _target[_target < eps] = eps

    ratio = _target / _input
    loss = ratio - np.log(ratio) - 1

    return loss

def generalized_kl_divergence(input, target, eps=EPS):
    """
    Args:
        input (...)
    """
    _input, _target = input.copy(), target.copy()
    _input[_input < eps] = eps
    _target[_target < eps] = eps

    ratio = _target / _input
    loss = _target * np.log(ratio) + _input - _target

    return loss

def beta_divergence(input, target, beta=2):
    """
    Beta divergence

    Args:
        input (batch_size, ...)
    """
    beta_minus1 = beta - 1

    assert beta != 0, "Use is_divergence instead."
    assert beta_minus1 != 0, "Use generalized_kl_divergence instead."

    loss = target * (target**beta_minus1 - input**beta_minus1) / beta_minus1 - (target**beta - input**beta) / beta
    
    return loss