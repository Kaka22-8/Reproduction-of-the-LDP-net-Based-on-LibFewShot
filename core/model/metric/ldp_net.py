# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel

class ProtoHead(nn.Module):
    """
    LDP-Net uses a simple ProtoNet head in training:
    support prototype vs. query features with negative euclidean distance.
    """
    def __init__(self):
        super(ProtoHead, self).__init__()

    def forward(self, proto, query):
        n = query.size(0)
        m = proto.size(0)
        d = query.size(1)
        assert d == proto.size(1)

        query = query.unsqueeze(1).expand(n, m, d)
        proto = proto.unsqueeze(0).expand(n, m, d)
        scores = -torch.pow(query - proto, 2).sum(2)
        return scores

class LDPNet(MetricModel):
    def __init__(
            self,
            feat_dim=512,
            lambda1 = 1.0,
            lambda2 = 0.15,
            m=0.998,
            beta=0.5,
            **kwargs
    ):
        super(LDPNet,self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.m = m
        self.beta = beta

        self.loss_func = nn.CrossEntropyLoss()
        self.proto_head = ProtoHead()

        self.local_branch = copy.deepcopy(self.emb_func)
        for param in self.local_branch.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_local_branch(self):
        for param_q, param_k in zip(self.emb_func.parameters(), self.local_branch.parameters()):
            param_k.data=param_k.data * self.m + param_q.data * (1.0 - self.m)

    def train(self, mode=True):
        super(LDPNet,self).train(mode)
        self.local_branch.train(False)
        return self

    def set_forward(self,batch):
        """
        Official-style test path:
        1. extract support/query features with emb_func
        2. fit LogisticRegression on support
        3. predict on query
        """
        image,global_target=batch
        image=image.to(self.device)

        with torch.no_grad():
            feat=self.emb_func(image)
            feat=torch.pow(feat.clamp(min=1e-6),self.beta)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat,mode=1
        )
        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []

        for idx in range(episode_size):
            sf = support_feat[idx]
            qf = query_feat[idx]
            st = support_target[idx]
            qt = query_target[idx]

            classifier= self.set_forward_adaptation(sf,st)

            qf = qf.detach().cpu().numpy()
            qt=qt.detach().cpu().numpy()

            pred = classifier.predict(qf)
            acc = metrics.accuracy_score(qt,pred) * 100.0

            output_list.append(pred)
            acc_list.append(acc)

        output=np.stack(output_list,axis=0)
        acc = sum(acc_list)/episode_size

        return output, acc

    def set_forward_loss(self, batch):
        """
        LDP-Net training path:
        - raw/global branch provides support prototypes and query predictions
        - EMA local branch provides local-view teacher targets
        - losses: ce + lambda1 * self-image + lambda2 * cross-image
        """
        batch_views, global_labels = batch

        raw_images = batch_views["raw_images"].to(self.device)
        local_views = [view.to(self.device) for view in batch_views["local_views"]]

        episode_size, way_num, total_num, c, h, w = raw_images.size()
        assert way_num == self.way_num
        assert total_num == self.shot_num + self.query_num

        raw_feat = self.emb_func(raw_images.view(-1, c, h, w))
        raw_feat = raw_feat.view(episode_size, way_num, total_num, -1)

        query_target = (
            torch.arange(way_num, dtype=torch.long, device=self.device)
            .view(1, way_num, 1)
            .repeat(episode_size, 1, self.query_num)
            .view(episode_size, -1)
        )

        ce_loss = 0.0
        self_image_loss = 0.0
        cross_image_loss = 0.0
        output_list = []
        acc_list = []

        # Precompute local teacher features once for all local views.
        with torch.no_grad():
            local_feat_list = []
            for local_view in local_views:
                local_feat = self.local_branch(local_view.view(-1, c, 96, 96))
                local_feat = local_feat.view(episode_size, way_num, total_num, -1)
                local_feat_list.append(local_feat)

        for epi in range(episode_size):
            support_feat = raw_feat[epi, :, : self.shot_num, :]
            query_feat = raw_feat[epi, :, self.shot_num :, :]

            proto = support_feat.mean(dim=1)
            query_feat_flat = query_feat.contiguous().view(
                way_num * self.query_num, -1
            )

            output = self.proto_head(proto, query_feat_flat)
            ce_loss = ce_loss + self.loss_func(output, query_target[epi])
            acc_list.append(accuracy(output, query_target[epi]))
            output_list.append(output)

            global_prob = F.softmax(output, dim=-1).detach()

            global_prob_4d = global_prob.view(way_num, self.query_num,way_num)
            if self.query_num > 1:
                cross_global_prob_4d = (
                    global_prob_4d.sum(dim=1,keepdim=True)-global_prob_4d
                ) / (self.query_num-1)
            else:
                cross_global_prob_4d = global_prob_4d

            cross_global_prob = cross_global_prob_4d.view(
                way_num *self.query_num,way_num
            )

            for local_feat in local_feat_list:
                local_query_feat = local_feat[epi, :, self.shot_num :, :].contiguous().view(
                    way_num * self.query_num, -1
                )
                local_output = self.proto_head(proto, local_query_feat)
                local_log_prob = F.log_softmax(local_output, dim=-1)

                self_image_loss = self_image_loss + F.kl_div(
                    local_log_prob,
                    global_prob,
                    reduction="batchmean",
                )
                cross_image_loss = cross_image_loss + F.kl_div(
                    local_log_prob,
                    cross_global_prob,
                    reduction="batchmean",
                )

        ce_loss = ce_loss / episode_size
        denom = episode_size * len(local_views)
        self_image_loss = self_image_loss / denom
        cross_image_loss = cross_image_loss / denom

        loss = (
            ce_loss
            + self.lambda1 * self_image_loss
            + self.lambda2 * cross_image_loss
        )
        output = torch.cat(output_list, dim=0)
        acc = sum(acc_list) / episode_size

        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(
            penalty="l2",
            random_state=0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
        )

        support_feat = support_feat.detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)
        return classifier
