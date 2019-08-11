"""Functions for measuring sensitivity and generalisation in FC neural networks.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: November 2018

Notes
-----

Based on the "Jacobian norm" sensitivity metric proposed by [1]_.

References
----------
.. [1] R. Novak, Y. Bahri, D. A. Abolafia, J. Pennington, J. Sohl-Dickstein (2018):
    Sensitivity and Generalization in Neural Networks: an Empirical Study.
        https://arxiv.org/abs/1802.08760
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch


def input_output_jacobian(
        inputs: torch.Tensor,
        outputs: torch.Tensor) -> torch.Tensor:
    """Compute the input-output Jacobian matrix for some function.

    Notes
    -----
    Based on https://discuss.pytorch.org/t/calculating-jacobian-in-a-differentiable-way/13275/4

    Parameters
    ----------
    inputs : torch.Tensor
        Input vector used to produce outputs.
    outputs : torch.Tensor
        Output vector of a function on inputs.

    Returns
    -------
    jacobian : torch.Tensor
        The Jacobian matrix of the function outputs w.r.t. inputs.
    """
    if inputs.grad is not None:
        inputs.grad.zero()  # manually zero gradients before computing jacobian
    jacobian = [torch.autograd.grad([outputs[:, i].sum()],
                                    [inputs],
                                    retain_graph=True,
                                    create_graph=True)[0]
                for i in range(outputs.size(1))]
    # print(jacobian)
    return torch.stack(jacobian, dim=-1)


def frobenius_norm(jacobian: torch.Tensor) -> torch.Tensor:
    """[summary]

    [description]

    Parameters
    ----------
    jacobian : torch.Tensor
        [description]

    Returns
    -------
    frobenius_norm: torch.Tensor
        [description]
    """
    return torch.norm(jacobian, dim=1)
    # return jacobian.pow(2).view(jacobian.size(0), -1).sum(dim=1).pow(0.5)


def local_sensitivity(x_test: torch.Tensor,
                      model: torch.nn.Module,
                      logits_only=False) -> torch.Tensor:
    """[summary]

    [description]

    Parameters
    ----------
    x_test : torch.Tensor
        [description]
    model : torch.nn.Module
        [description]

    Returns
    -------
    torch.Tensor
        [description]
    """
    x_test.requires_grad_(requires_grad=True)  # record ops on this tensor
    outputs = model(x_test)
    if not logits_only:
        outputs = torch.softmax(outputs, dim=-1)
    jacobian_norm = frobenius_norm(input_output_jacobian(inputs=x_test,
                                                         outputs=outputs))

    return jacobian_norm.mean()
