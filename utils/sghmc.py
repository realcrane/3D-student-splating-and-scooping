import torch
import numpy as np
from torch.optim import Optimizer
import math



class AdamSGHMC(Optimizer):
    name = "SGHMC"
    def __init__(self,
                 params,
                 lr: float=1e-2,
                 betas = (0.9, 0.999),
                 eps = 1e-8,
                 weight_decay = 0,
                 mdecay: float=0.01,
                 wd: float=0.00002,
                 scale_grad: float=1.,
                 mdecay_burnin: float=0.01,
                 burnin_iterations: int=7000) -> None:
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mdecay=mdecay,
            wd=wd,
            scale_grad=scale_grad,
            mdecay_burnin=mdecay_burnin,
            burnin_iterations=burnin_iterations
        )
        super().__init__(params, defaults)

        self.pre_momentum = []

    def step(self, closure=None, sig=None, cov=None):
        loss = None

        if closure is not None:
            loss = closure()

        optim_xyz = False
        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                grad = parameter.grad.data
                state = self.state[parameter]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    state["iteration"] = 0
                    state["momentum"] = grad.new().resize_as_(grad).zero_()

                state["iteration"] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], parameter.data)
                
                 # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = math.sqrt(bias_correction2) / bias_correction1

                base = torch.zeros_like(exp_avg)
                adam_gradient = torch.addcdiv(base, exp_avg, denom, value=-step_size)
                if group['name'] != 'xyz':
                    parameter.data.add_(group['lr'] * adam_gradient)
                    continue

                # NOTE: SGHMC
                optim_xyz = True
                mdecay, lr, wd = group["mdecay"], group["lr"], group["wd"]
                mdecay_burnin, burnin_iterations = group["mdecay_burnin"], group["burnin_iterations"]
                scale_grad = group["scale_grad"]

                if state["iteration"] == burnin_iterations+1:
                    state["momentum"] = grad.new().resize_as_(grad).zero_()
                    
                momentum = state["momentum"]
                last_momentum = momentum.detach().clone()
                gradient = -adam_gradient * scale_grad
                
                # NOTE: remove friction for burn in
                if state["iteration"] <= burnin_iterations:
                    sigma = torch.sqrt(torch.from_numpy(np.array(2 * lr * mdecay_burnin, dtype=type(lr))))
                    sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)
                    noise_t = sample_t * sig
                    noise_t = torch.bmm(cov, noise_t.unsqueeze(-1)).squeeze(-1)
                    friction = lr * (1/lr) * momentum * sig
                    momentum.mul_(sig)
                    momentum.add_(-lr * gradient - friction + noise_t)
                else:
                    # NOTE: sigmoid is applied to total noise
                    sigma = torch.sqrt(torch.from_numpy(np.array(2 * lr * mdecay, dtype=type(lr))))
                    sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)
                    noise_t = sample_t * sig
                    friction = lr * mdecay * momentum * sig
                    momentum.mul_(sig)
                    momentum.add_(-lr * gradient - friction + noise_t)

                parameter.data.add_(lr * momentum)


        if not optim_xyz:
            return loss, torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1))
        else:
            if state["iteration"] > 2 and state["iteration"] <= burnin_iterations:
                return loss, (lr * (1-lr*(1/lr)) * last_momentum + lr * sample_t).max(), (lr * (1-lr*(1/lr)) * last_momentum + lr * sample_t).mean(), (lr * (1-lr*(1/lr)) * last_momentum + lr * sample_t).min()
            elif state["iteration"] > burnin_iterations:
                return loss, (lr * (1-lr*mdecay) * last_momentum + lr * sample_t).max(), (lr * (1-lr*mdecay) * last_momentum + lr * sample_t).mean(), (lr * (1-lr*mdecay) * last_momentum + lr * sample_t).min()
            else:
                return loss, torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1))