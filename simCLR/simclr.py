import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#from utils import save_config_file, accuracy, save_checkpoint
import torch.nn as nn
torch.manual_seed(0)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature


    # def forward(self, z_i, z_j):
    #     batch_size = z_i.shape[0]
    #     z = torch.cat((z_i, z_j), dim=0)
    #     sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
    #     mask = torch.eye(batch_size * 2, device=z.device, dtype=torch.bool)
    #     sim_matrix = sim_matrix[~mask].view(batch_size * 2, -1)
    #     sim_matrix = torch.cat((sim_matrix, sim_matrix), dim=0)
    #     mask = torch.cat((torch.ones((batch_size, batch_size)), torch.zeros((batch_size, batch_size))), dim=1).bool()
    #     print(mask.shape)
    #     print(mask.t().shape)
    #     mask = ~torch.logical_or(mask, mask.t())
    #     loss = -torch.log(sim_matrix[mask]/torch.sum(sim_matrix, dim=1))
    #     return loss.mean()

    # def forward(self, z_i, z_j):
    #     batch_size = z_i.shape[0]
    #     z = torch.cat((z_i, z_j), dim=0)
    #     sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
    #     mask = torch.eye(batch_size * 2, device=z.device, dtype=torch.bool)
    #     mask = mask
    #     mask = ~mask
    #     mask = mask.expand(batch_size * 2, batch_size * 2)
    #     sim_matrix = sim_matrix.masked_select(mask).view(batch_size * 2, -1)
    #
    #     pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
    #     neg_sim = torch.exp(torch.mm(z_i, z_j.t().contiguous()) / self.temperature)
    #     neg_sim = neg_sim.view(batch_size * 2, -1)
    #
    #     pos_sim = pos_sim.unsqueeze(-1)
    #     neg_sim = neg_sim.mean(dim=-1, keepdim=True)
    #     pos_sim = torch.cat((pos_sim, pos_sim), dim=0)
    #
    #
    #     print(pos_sim.shape)
    #     print(neg_sim.shape)
    #     loss = -(torch.log(pos_sim / torch.sum(sim_matrix, dim=-1, keepdim=True)) +
    #              torch.log(torch.sum(neg_sim, dim=-1, keepdim=True) / torch.sum(sim_matrix, dim=-1, keepdim=True)))
    #     return loss.mean()

    # def forward(self, z_i, z_j):
    #     batch_size = z_i.shape[0]
    #     z = torch.cat((z_i, z_j), dim=0)
    #     sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
    #     mask = torch.eye(batch_size * 2, device=z.device, dtype=torch.bool)
    #     mask = ~mask
    #     mask = mask.expand(batch_size * 2, batch_size * 2)
    #     sim_matrix = sim_matrix.masked_select(mask).view(batch_size * 2, -1)
    #
    #     pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
    #     neg_sim = torch.exp(torch.mm(z_i, z_j.t().contiguous()) / self.temperature)
    #     neg_sim = torch.cat((neg_sim, neg_sim), dim=0)
    #
    #
    #     pos_sim = pos_sim.unsqueeze(-1)
    #     neg_sim = neg_sim.view(batch_size, -1)
    #     logits = torch.cat((pos_sim, neg_sim), dim=-1)
    #     labels = torch.zeros(batch_size, dtype=torch.long).to(z_i.device)
    #     loss = nn.CrossEntropyLoss()(logits / self.temperature, labels)
    #
    #     return loss

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        self.batch_size = emb_i.shape[0]

        self.register_buffer("negatives_mask", (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)


        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature).to(self.negatives_mask.device)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1).to(nominator.device))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_size=128):
        super(SimCLR, self).__init__()
        self.base_encoder = base_encoder
        self.projection_size = projection_size
        self.encoder = nn.Sequential(
            nn.Linear(self.base_encoder.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.projection_size)
        )

    def forward(self, x):
        h = self.base_encoder(x)
        z = self.encoder(h)
        return z
