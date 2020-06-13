from torch.optim import Optimizer
from torch.optim.optimizer import required


class START(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0, mu=1):
        self.lr = lr
        self.mu = mu
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(START, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(START, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        ###########################################
        loss = None
        mu = self.mu
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            # c1 = mu/(lr+mu)
            # c2 = lr/mu
            c = lr*mu
            for p, pc in zip(group['params'], group['prox_center']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # p.data.add_(c2, pc)
                p.data.mul_(1-c)
                p.data.add_(-lr, d_p)
                p.data.add_(c, pc)
                # p.data.mul_(c1)

        # p.data <- (mu * p.data + lr * pc - lr * mu * d_p) / (lr + mu)

        ###########################################
        return loss

    def adjust_learning_rate(self, round_i):
        lr = self.lr * (0.5 ** (round_i // 30))
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def soft_decay_learning_rate(self):
        self.lr *= 0.99
        for param_group in self.param_groups:
            param_group['lr'] = self.lr

    def inverse_prop_decay_learning_rate_exp(self, round_i):
        for param_group in self.param_groups:
            #param_group['lr'] = self.lr/(round_i+1)
            param_group['lr'] =self.lr * (0.5 ** (round_i))

    def inverse_prop_decay_learning_rate(self, round_i):
        for param_group in self.param_groups:
            param_group['lr'] = self.lr/(round_i+1)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']

    def update_prox_center(self):
        for group in self.param_groups:
            saved_param = []
            for p in group['params']:
                p_copy = p.data.clone().detach()
                saved_param.append(p_copy)

            group['prox_center'] = saved_param


class GD(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        self.lr = lr
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(GD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], d_p)
        return loss

    def adjust_learning_rate(self, round_i):
        lr = self.lr * (0.5 ** (round_i // 30))
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def soft_decay_learning_rate(self):
        self.lr *= 0.99
        for param_group in self.param_groups:
            param_group['lr'] = self.lr

    def inverse_prop_decay_learning_rate(self, round_i):
        for param_group in self.param_groups:
            param_group['lr'] = self.lr/(round_i+1)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']


class LrdGD(GD):
    def __init__(self, params, lr=required, weight_decay=0):
        super(LrdGD, self).__init__(params, lr, weight_decay)

    def step(self, lr, closure=None):
        """Performs a single optimization step.

        Arguments:
            lr: learning rate
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-lr, d_p)
        return loss
