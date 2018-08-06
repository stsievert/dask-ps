from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
import copy


def clone(model):
    return copy.deepcopy(model)


class PS:
    def __init__(self, model, args, device, num_workers=1):
        self.models = {0: model}
        self._grads = {}
        model = model.to(device)
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        self.num_workers = num_workers
        
    def pull(self, key):
        """
        For a worker to pull a model from this PS
        """
        if key not in self.models:
            return None
        return self.models[key]
    
    @property
    def step(self):
        return max(self.models)
    
    @property
    def latest_model(self):
        return self.models[self.step]
    
    def pull_latest(self):
        step = max(self.models)
        return step, self.pull(step)
        
    def push(self, key, grads):
        """
        For a worker to push some gradients to this PS
        """
        if key not in self._grads:
            self._grads[key] = []
        self._grads[key] += [grads]
        
        # have we collected enough gradients?
        if len(self._grads[key]) == self.num_workers:
            old_model = clone(self.model)
            self.aggregate(self._grads[key])
            self.models[key + 1] = self.model
            self.models[key] = old_model
    
    def aggregate(self, grads):
        for name, param in self.model.named_parameters():
            worker_grads = [grad[name] for grad in grads]
            param.grad = sum(worker_grads)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        
class Worker:
    def __init__(self, model,
                 args, worker_id=0, num_workers=1, batch_size=64,
                 compute_grad=None):
        assert compute_grad is not None
        self.compute_grad = compute_grad
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        
        torch.manual_seed(args.seed + worker_id)
        
        batch_size //= num_workers  
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
        batch_size=batch_size, shuffle=True, **kwargs)
        self.train_loader = train_loader
        self.args = args
        
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        
        self.grads = []
        self.worker_id = worker_id
        self.num_workers = num_workers
        
        self.train_loader = train_loader
        
    @property
    def metadata(self):
        return self.meta
    
    @property
    def _model(self):
        return self.model
    
    def compute_gradients(self):
        data, target = next(iter(self.train_loader))
        self.grad = self.compute_grad(self.model, data, target, self.optimizer,
                                 self.args, self.device)
        return True
    
    def send(self, worker):
        worker.recv(self.grad)
        return True
        
    def recv(self, grad):
        self.grads += [grad]
        return True
    
    def apply_gradients(self):
        # This is a little verbose; no all-reduce, list of grads, etc
        if len(self.grads) != self.num_workers:
            return False
            
        grads = {k: sum([named_grads[k] for named_grads in self.grads])
                for k in self.grads[0]}
        self.grads = []
        
        for name, param in self.model.named_parameters():
            param.grad = grads[name]
        check = param.detach().numpy().flat[:3]
        if self.worker_id == 0:
            print('-' * 40)
        print(self.worker_id, check)
            
        self.optimizer.step()
        return True