"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss, smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.rboxlist_ops import targets_for_locations
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import time

INF = 100000000

# def onehot(label, cls_num):
#     onehot = np.zeros(self.__num_classes, dtype=np.float)
#     onehot[bbox_class_ind] = 1.0
#     uniform_distribution = np.full(self.__num_classes, 1.0 / self.__num_classes)
#     deta = 0.01
#     smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # self.cls_loss_func = nn.CrossEntropyLoss(size_average=False, reduce=True)

        self.cfg = cfg
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        # self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.num_pts = cfg.MODEL.FCOS.NUM_PTS

    # 2
    def prepare_targets(self, points, targets):
        # FoveaBox
        # strides=[8, 16, 32, 64, 128],
        # base_edge_list=[16, 32, 64, 128, 256],
        # scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        if self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "fcos":
            # FCOS
            object_sizes_of_interest = [
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, INF],
            ]
            # object_sizes_of_interest = [
            #     [-1, 32],
            #     [32, 64],
            #     [64, 128],
            #     [128, 256],
            #     [256, INF],
            # ]
        elif self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "foveabox":
            object_sizes_of_interest = [
                [-1, 64],
                [32, 128],
                [64, 256],
                [128, 512],
                [256, INF],
            ]
        elif self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "all":
                object_sizes_of_interest = [
                [-1, 64],
                [-1, 128],
                [-1, 256],
                [-1, 512],
                [-1, INF],
            ]
        normal_factor = [16, 32, 64, 128, 256]
        # normal_factor = [16, 48, 96, 192, 384]


        expanded_object_sizes_of_interest = []
        expanded_normal_factor=[]
        # p3 - p7
        for l, points_per_level in enumerate(points):
            # 2
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            # 1 2 -> len(points_per_level) 2
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
            normal_factor_per_level = \
                points_per_level.new_tensor(normal_factor[l])
            # 1 2 -> len(points_per_level) 2
            expanded_normal_factor.append(
                normal_factor_per_level.expand(len(points_per_level))
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        expanded_normal_factor = torch,cat(expanded_normal_factor, dim=0)

        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        # batch len(locations) 1   batch len(locations) 6 
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, expanded_normal_factor
        )

        # 对每一张图片进行处理
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first


    # 3
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest, normal_factor):
        labels = []
        reg_targets = []
        # xs, ys = locations[:, 0], locations[:, 1]
        
        for im_i in range(len(targets)):
            # 第i张图片
            targets_per_im = targets[im_i]
            # assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")#.cpu()
            # print(labels_per_im)

           
            reg_targets_per_im = targets_for_locations(bboxes, locations)#.cpu()
            # torch.cuda.empty_cache()

            # max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            
            max_reg_targets_per_im = torch.abs(reg_targets_per_im[:,2:6]).max(dim=1)[0]
            # distance
            
            dist_1 = torch.sqrt(torch.pow(reg_targets_per_im[:,2],2) + torch.pow(reg_targets_per_im[:,3],2))
            dist_2 = torch.sqrt(torch.pow(reg_targets_per_im[:,4],2) + torch.pow(reg_targets_per_im[:,5],2))
            target_h = reg_targets_per_im[:,5] 
            max_reg_targets_per_im = torch.stack([dist_1, dist_2, target_h], dim=1).max(dim=1)[0]

            # limit the regression range for each location 上下左右都要在感兴趣范围之内

            # len(locations)  len(locations), 1
            object_sizes_of_interest= object_sizes_of_interest#.cpu()
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, 0]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, 1])
            
            # print("labels_per_im", len(labels_per_im), len(bboxes), torch.min(reg_targets_per_im[:, 0].long()), torch.max(reg_targets_per_im[:, 0].long()), reg_targets_per_im[:, 0].sum())
            labels_per_im = labels_per_im[reg_targets_per_im[:, 0].long()]

            # 落在目标框外面label为0
            labels_per_im[reg_targets_per_im[:, 1] < 0.5 ] = 0#bg
            # 或者落在外面且s感受野不够
            labels_per_im[is_cared_in_the_level == 0] = 0#no reg

            # 落在框内 但是感受野不够
            labels_per_im[(reg_targets_per_im[:, 1] > 0.5) * (is_cared_in_the_level == 0)] = -1#ignore

            # detax1 detay1  detax2 detay2 h
            ones = torch.ones_like(reg_targets_per_im[:,2:7])
            one_minusone = torch.where(reg_targets_per_im[:,2:7]>=0, ones, -ones)#.cpu()
    
            reg_targets_per_im[:,2:7] = one_minusone*torch.pow(torch.abs(reg_targets_per_im[:,2:7])/normal_factor[1][:,None], 1/3)#.cpu()#.cpu()

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im[:,2:])

        return labels, reg_targets 

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    # 1
    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # 0 fpn 第一层
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)#//self.num_pts

        # level first
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        # for level
        for l in range(len(labels)):
            # batch*num_pos num_classes
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 5))
            # layer_h, layer_w = box_cls[l].size(2), box_cls[l].size(3)
            # box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(N, layer_h, layer_w, self.num_pts, num_classes).permute(0, 3, 1, 2, 4).reshape(-1,num_classes))
            # box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(N, layer_h, layer_w, self.num_pts, 5).permute(0, 3, 1, 2, 4).reshape(-1,5))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 6))
            centerness_flatten.append(centerness[l].reshape(-1))
        # level batch*num_pos num_classes
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds   = torch.nonzero(labels_flatten > 0).squeeze(1)
        valid_inds = torch.nonzero(labels_flatten > -1).squeeze(1)
        ignore_inds = torch.nonzero(labels_flatten == -1).squeeze(1)

        # wrong
        # cls_weight=torch.where(centerness_flatten==0, torch.ones_like(centerness_flatten), centerness_flatten).unsqueeze(-1)
        # cls_loss = self.cls_loss_func(
        #     box_cls_flatten,#.cpu()
        #     labels_flatten.int(),#,#.cpu()
        #     weight = cls_weight
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        # true
        all_centerness_targets = reg_targets_flatten[:, -1]
        # # torch.sqrt(
        cls_weight = torch.where(all_centerness_targets==0, torch.ones_like(all_centerness_targets), all_centerness_targets).unsqueeze(-1)
        cls_weight[ignore_inds] = 0.05
        # # cls_weight=torch.where(all_centerness_targets==0, torch.full_like(all_centerness_targets, 1), all_centerness_targets).unsqueeze(-1)

        ''' 涉及到将感受野不够或者超过的点看做负样本/忽略样本/正样本
        看成忽略样本涉及到定位不准确　冗余检测
        看成负样本 conf is  low'''
        # # focal loss 2*
        # cls_loss = self.cls_loss_func(
        #     box_cls_flatten[valid_inds],#.cpu()
        #     labels_flatten[valid_inds].int(),#.cpu()
        #     weight = cls_weight[valid_inds]
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        
        
        # weight = cl
        labels_flatten[labels_flatten==-1]=0
        cls_loss = self.cls_loss_func(
            box_cls_flatten,#.cpu()
            labels_flatten.int(),#.cpu()
            weight = cls_weight      ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        # s_weight, 
        # self.cls_loss_func = nn.CrossEntropyLoss(size_average=False, reduce=True)
        # cls_loss = self.cls_loss_func(
        #     box_cls_flatten[pos_inds],#
        #     labels_flatten[pos_inds]-1,#
        # )/ (pos_inds.numel() + N)


        box_regression_pos = box_regression_flatten[pos_inds]
        reg_targets_pos = reg_targets_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets_pos = reg_targets_pos[:, -1]

            #只预测
            reg_loss = smooth_l1_loss(
                box_regression_pos,#.cpu()
                reg_targets_pos[:, :-1],#.cpu()
                weight = centerness_targets_pos.unsqueeze(-1)#cls_weight #
            )

            # 一定要回归center ness
            # all 
            # centerness_flatten[ignore_inds] = 0
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,#.cpu()
                reg_targets_flatten[:,-1]#.cpu()
            )
            # centerness_loss = self.centerness_loss_func(
            #     centerness_flatten[pos_inds],#.cpu()
            #     centerness_targets_pos#.cpu()
            # )

        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            # .cuda()
        return cls_loss, reg_loss, centerness_loss#*0


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
