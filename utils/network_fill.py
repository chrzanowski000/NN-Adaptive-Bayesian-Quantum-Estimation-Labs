import torch
import torch.nn.init as init

def fill_policy_gaussian(policy, mu, sigma):
    """
    Fill all parameters of a policy network with samples
    from N(mu, sigma^2).

    Args:
        policy (torch.nn.Module): TimePolicy instance
        mu (float): mean of Gaussian
        sigma (float): std of Gaussian
    """
    with torch.no_grad():
        for p in policy.parameters():
            init.normal_(p, mean=mu, std=sigma)