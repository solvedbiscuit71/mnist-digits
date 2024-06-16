#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:01:36 2024

@author: solvedbiscuit71
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_loader import training_data, test_data

if not torch.backends.mps.is_available():
    print ("MPS device not found.")

class Shallow300(nn.Module):
    def __init__(self, random_state=47):
        super().__init__()
        torch.manual_seed(random_state)
        self.W0 = nn.Parameter(torch.randn(28*28, 300).mul_(2/pow(28*28, 0.5)))
        self.b0 = nn.Parameter(torch.zeros(300))

        self.W1 = nn.Parameter(torch.randn(300, 10).mul_(1/pow(300, 0.5)))
        self.b1 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = torch.matmul(x, self.W0) + self.b0
        x = F.relu(x)
        x = torch.matmul(x, self.W1) + self.b1
        return x

def loss_fn(logit, y):
    log_prob = F.log_softmax(logit, dim=1)
    return torch.mean(-torch.sum(y * log_prob, dim=1))

class AdamOptimizer:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, device="cpu"):
        self.lr = lr
        self.beta1 = torch.tensor(beta1, device=device)
        self.beta2 = torch.tensor(beta2, device=device)
        self.epsilon = torch.tensor(epsilon, device=device)

        self.parameters = list(parameters)
        self.v = [torch.zeros_like(param, device=device) for param in self.parameters]
        self.s = [torch.zeros_like(param, device=device) for param in self.parameters]

    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * param.grad
                self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * param.grad ** 2
                param -= self.lr * self.v[i] / torch.sqrt(self.s[i] + self.epsilon)

    def zero_grad(self):
        for param in self.parameters:
            param.grad.zero_()

def train_loop(dataset, epoch=50, lr=1e-3, *, m=128, beta1=0.9, beta2=0.999, device="cpu"):
    training_dataloader = DataLoader(dataset, batch_size=m, shuffle=True)
    model = Shallow300()
    model.to(device)
    optimizer = AdamOptimizer(model.parameters(), lr, beta1, beta2)

    losses = []

    try:
        for i in range(1, epoch+1):
            run_loss = []

            print(f"Epoch {i}: [", end="")
            for i, (images, labels) in enumerate(training_dataloader):
                images = images.to(device=device)
                labels = labels.to(device=device)
                logit = model(images)
                loss = loss_fn(logit, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                run_loss.append(loss.item())

                if (i+1) % 20 == 0:
                    print("=", end="")

            loss = sum(run_loss) / len(run_loss)
            losses.append(loss)
            print(f"] loss={loss}")

    except KeyboardInterrupt:
        print("Stopping training...")

    return model, losses

def evaluate(model, dataset, device="cpu"):
    model.eval()
    with torch.no_grad():
        test_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        total = 0
        count = 0
        for images, labels in test_dataloader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            logit = model(images)
            prob = F.softmax(logit, dim=1)
            count += torch.sum(labels.argmax(1) == prob.argmax(1)).item()
            total += len(labels)

    print(f"{count} out of {total}")
    return count / total * 100

model, losses = train_loop(training_data, epoch=40, lr=1e-3, device='mps')
print("Training Accuracy:", evaluate(model, training_data, device='mps'))
print("Test Accuracy:", evaluate(model, test_data, device='mps'))

torch.save(model.state_dict(), 'model/model_shallow_300hu.pth')
