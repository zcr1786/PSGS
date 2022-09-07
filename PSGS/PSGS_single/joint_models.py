import copy
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from torch_geometric.nn import SAGPooling, GCNConv
import pickle
import random
import numpy as np


import time
from tqdm import tqdm

import graph_models
import seq_models
import models
# from attention import SimplifiedScaledDotProductAttention
from attention import EMSA
class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese,self).__init__()
        self.siamese_text= nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64))
        self.siamese_image= nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64))
    def forward(self,text_feature,image_feature):
        output1 = self.siamese_text(text_feature)
        output2 = self.siamese_image(image_feature)

        output11 = F.normalize(torch.mean(output1, dim=0, keepdim=True), p=2, dim=1)
        output22 = F.normalize(torch.mean(output2, dim=0, keepdim=True), p=2, dim=1)
        return output11,output22

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        # cos = torch.cosine_similarity(output1, output2).sigmoid()
        # # print('cos' , cos)
        # loss_contrastive = (1-label) * (1- cos) + label * cos
        # # criterion = nn.CrossEntropyLoss()
        # # loss_contrastive = criterion(cos , label)

        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        xxl = torch.sigmoid(euclidean_distance)
        loss_contrastive = torch.mean((1-label) * torch.pow(xxl, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - xxl, min=0.0), 2))
        # loss_contrastive = torch.mean((1 - label) * xxl +
        #                               (label) * torch.clamp(self.margin - xxl, min=0.0))
        return loss_contrastive

class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()
        # self.ratio = 1.0
        # self.pool1 = SAGPooling(64, ratio=self.ratio, GNN=GCNConv)
        self.device = args.device
        self.layers = args.num_layers
        self.input_size_graph = args.input_size_graph
        self.output_size_graph = args.output_size_graph
        self.train_data = args.train_data
        self.test_data = args.test_data
        self.train_labels = args.train_labels
        self.test_labels = args.test_labels
        self.latent_size = args.latent_size
        self.hidden_size = args.hidden_size
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.warmup = args.warmup

        # self.att = SimplifiedScaledDotProductAttention(d_model=128, h=4)
        self.att = EMSA(d_model=128, d_k=128, d_v=128, h=10,H=10,W=10,ratio=2,apply_transform=True)
        self.graph = args.graph
        self.sequence = args.sequence
        self.recons = args.recons
        self.use_attn = args.attn
        self.use_fusion = args.fusion

        self.graph_pretrain = graph_models.GraphSage(self.layers,
                                                     self.input_size_graph,
                                                     self.output_size_graph,
                                                     device=self.device,
                                                     gcn="True",
                                                     agg_func="MEAN")

        self.VAE = seq_models.VAE(args)

        self.AtomEmbedding = nn.Embedding(self.input_size_graph,self.hidden_size).to(self.device)
        self.AtomEmbedding.weight.requires_grad = True

        self.output_layer = models.classifier(self.latent_size * 2, self.device)

        self.siamese = Siamese()
        self.sia_loss = ContrastiveLoss()
        self.label_criterion = nn.CrossEntropyLoss()

        if self.use_attn:
            self.attention = models.SelfAttention(self.hidden_size)

        self.optimizer  = optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-8,
                                     amsgrad=True)

        for name, para in self.named_parameters():
            if para.requires_grad:
                print(name, para.data.shape)

    def train(self, graph_index_x, graph_index_y, epoch):

        nodes_emb_x = self.AtomEmbedding(self.train_data['sequence'][graph_index_x])
        nodes_emb_y = self.AtomEmbedding(self.train_data['sequence'][graph_index_y])


        if self.graph:
            nodes_emb_x = self.graph_pretrain(graph_index_x, self.train_data)
            graph_emb_x = nodes_emb_x
            nodes_emb_y = self.graph_pretrain(graph_index_y, self.train_data)
            graph_emb_y = nodes_emb_y
        if self.sequence:
            recons_loss, nodes_emb_x = self.VAE(nodes_emb_x, epoch)
            seq_emb_x = nodes_emb_x
            recons_loss, nodes_emb_y = self.VAE(nodes_emb_y, epoch)
            seq_emb_y = nodes_emb_y

        grax = graph_emb_x
        seqx = seq_emb_x
        gray = graph_emb_y
        seqy = seq_emb_y

        mygrax, myseqx = self.siamese(grax, seqx)
        mygray, siaseqx = self.siamese(gray, seqx)
        siagrax, myseqy = self.siamese(grax, seqy)
        graymy, seqymy = self.siamese(gray, seqy)

        label_x = torch.LongTensor([self.train_labels[graph_index_x]]).to(self.device)
        label_y = torch.LongTensor([self.train_labels[graph_index_y]]).to(self.device)
        asd = label_x - label_y
        if asd==0 :
            zxc = 0.4
        else:
            zxc = 0
        simloss1 = self.sia_loss(mygrax, myseqx, 1)
        simloss2 = self.sia_loss(mygray, siaseqx, zxc)
        simloss3 = self.sia_loss(siagrax, myseqy, zxc)
        simloss4 = self.sia_loss(graymy, seqymy, 1)

        # print("========simloss1, simloss2=========")
        # print(simloss1)
        # print(simloss2)

        xx_distance = F.pairwise_distance(mygrax, myseqx, keepdim=True, p=2)
        xy_distance = F.pairwise_distance(mygray, siaseqx, keepdim=True, p=2)
        # loss_siamese = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # print('xxx_distance = ', xx_distance, xy_distance)
        # print("--------------------------------", F.normalize(torch.mean(graph_emb_x, dim=0, keepdim=True), p=2, dim=1).shape    ,  mygrax.shape  )
        if self.use_fusion:
            # molecule_emb = F.normalize(torch.mean(graph_emb_x, dim=0, keepdim=True), p=2, dim=1) + F.normalize(
            #     torch.mean(seq_emb_x, dim=0, keepdim=True), p=2, dim=1)

            molecule_emb = torch.cat((F.normalize(torch.mean(graph_emb_x, dim=0, keepdim=True), p=2, dim=1) , F.normalize(
                torch.mean(seq_emb_x, dim=0, keepdim=True), p=2, dim=1)) , dim= 1 )
            # print(molecule_emb.shape)
            # molecule_emb = mygrax+ myseqx
        else:
            molecule_emb = torch.mean(nodes_emb_x, dim=0, keepdim=True)

        # print(molecule_emb)

        # # print('molecule_emb  ', molecule_emb.shape)
        # molecule_emb = molecule_emb.unsqueeze(1).repeat(1, 100, 1)
        # # print('molecule_emb  ', molecule_emb.shape)
        # molecule_emb = self.att(molecule_emb, molecule_emb, molecule_emb)
        # molecule_emb = torch.mean(molecule_emb, dim=1)
        #



        pred = self.output_layer(molecule_emb)
        label = torch.LongTensor([self.train_labels[graph_index_x]]).to(self.device)

        self.optimizer.zero_grad()
        loss_label = self.label_criterion(pred, label)

        lossim = ( simloss1 +( simloss2 + simloss3  )*0.5 )

        loss = loss_label  +( simloss1 +( simloss2 + simloss3  )*0.5 )
        # print(loss, loss_label, lossim)

        # loss =  simloss1 + simloss2

        loss.backward()
        self.optimizer.step()


        return loss ,molecule_emb

    def test(self, graph_index):

        nodes_emb = self.AtomEmbedding(self.test_data['sequence'][graph_index])

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_index, self.test_data)
            graph_emb = nodes_emb

        if self.sequence:
            nodes_emb = self.VAE.test_vae(nodes_emb)
            seq_emb = nodes_emb

        if self.use_fusion:
            # molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
            molecule_emb = torch.cat( (F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1), F.normalize(
                    torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)), dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)

        # molecule_emb = molecule_emb.unsqueeze(1).repeat(1, 100, 1)
        # # print('molecule_emb  ', molecule_emb.shape)
        # molecule_emb = self.att(molecule_emb, molecule_emb, molecule_emb)
        # molecule_emb = torch.mean(molecule_emb, dim=1)

        pred = self.output_layer(molecule_emb)

        return pred