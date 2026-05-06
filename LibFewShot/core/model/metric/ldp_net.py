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
        self.local_branch.train(mode)
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
        Official-style LDP-Net training path:
        - raw/global branch provides support/query anchor features
        - EMA branch provides local crop features
        - anchor and local queries are concatenated before one proto-head forward
        """
        if (
            isinstance(batch, (list, tuple))
            and len(batch) == 2
            and isinstance(batch[0], dict)
        ):
            batch_views, global_labels = batch
            raw_images = batch_views["raw_images"].to(self.device)
            local_views = [view.to(self.device) for view in batch_views["local_views"]]
        else:
            # Official-style per-class sub-dataloader path:
            # batch = [global1, global2, local1..local6, raw]
            assert isinstance(batch, (list, tuple)) and len(batch) >= 9
            raw_images = batch[8].to(self.device)
            local_views = [view.to(self.device) for view in batch[2:8]]

            if raw_images.dim() == 5:
                raw_images = raw_images.unsqueeze(0)
            local_views = [
                view.unsqueeze(0) if view.dim() == 5 else view for view in local_views
            ]

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

        # Local queries come from the EMA branch in the released official code.
        with torch.no_grad():
            local_feat_list = []
            for local_view in local_views:
                local_feat = self.local_branch(local_view.view(-1, c, 96, 96))
                local_feat = local_feat.view(episode_size, way_num, total_num, -1)
                local_feat_list.append(local_feat)

        for epi in range(episode_size):
            support_anchor = raw_feat[epi, :, : self.shot_num, :]
            query_anchor = raw_feat[epi, :, self.shot_num :, :]

            proto = support_anchor.mean(dim=1)
            query_anchor = query_anchor.contiguous().view(
                way_num * self.query_num, -1
            ).unsqueeze(0)

            local_query_all = []
            for local_feat in local_feat_list:
                local_query = local_feat[epi, :, self.shot_num :, :].contiguous().view(
                    way_num * self.query_num, -1
                )
                local_query_all.append(local_query)
            local_query_all = torch.stack(local_query_all, dim=0)

            query_set = torch.cat((query_anchor, local_query_all), dim=0)
            query_set = query_set.contiguous().view(
                (1 + len(local_views)) * way_num * self.query_num, -1
            )

            pred_query_set = self.proto_head(proto, query_set)
            pred_query_set = pred_query_set.contiguous().view(
                1 + len(local_views), way_num * self.query_num, way_num
            )

            pred_query_set_anchor = pred_query_set[0]
            pred_query_set_aug = pred_query_set[1:]

            ce_loss = ce_loss + self.loss_func(pred_query_set_anchor, query_target[epi])
            acc_list.append(accuracy(pred_query_set_anchor, query_target[epi]))
            output_list.append(pred_query_set_anchor)

            pred_query_set_anchor_prob = F.softmax(pred_query_set_anchor, dim=-1)
            pred_query_set_aug_prob = F.softmax(
                pred_query_set_aug.contiguous().view(
                    len(local_views) * way_num * self.query_num, way_num
                ),
                dim=-1,
            )

            pred_query_set_global = pred_query_set_anchor_prob.unsqueeze(0).repeat(
                len(local_views), 1, 1
            ).view(len(local_views) * way_num * self.query_num, way_num)

            self_image_loss = self_image_loss + torch.mean(
                torch.sum(
                    -pred_query_set_global
                    * torch.log(pred_query_set_aug_prob.clamp(min=1e-8)),
                    dim=1,
                )
            )

            pred_query_set_global = pred_query_set_anchor.view(
                way_num, self.query_num, way_num
            )
            rand_id_global = torch.randperm(self.query_num, device=self.device)[0]
            pred_query_set_global = pred_query_set_global[:, rand_id_global, :]
            pred_query_set_global = F.softmax(pred_query_set_global, dim=-1)
            pred_query_set_global = pred_query_set_global.unsqueeze(0).expand(
                len(local_views), way_num, way_num
            )
            pred_query_set_global = pred_query_set_global.contiguous().view(
                len(local_views) * way_num, way_num
            )

            rand_id_local = torch.randperm(self.query_num, device=self.device)[0]
            pred_query_set_local = pred_query_set_aug_prob.view(
                len(local_views), way_num, self.query_num, way_num
            )
            pred_query_set_local = pred_query_set_local[:, :, rand_id_local, :]
            pred_query_set_local = pred_query_set_local.contiguous().view(
                len(local_views) * way_num, way_num
            )

            cross_image_loss = cross_image_loss + torch.mean(
                torch.sum(
                    -pred_query_set_global
                    * torch.log(pred_query_set_local.clamp(min=1e-8)),
                    dim=1,
                )
            )

        ce_loss = ce_loss / episode_size
        self_image_loss = self_image_loss / episode_size
        cross_image_loss = cross_image_loss / episode_size

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
