import torch.nn as nn
import torch as th
import torch.nn.functional as F
import torch


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j
