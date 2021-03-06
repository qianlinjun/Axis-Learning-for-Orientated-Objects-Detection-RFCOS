import math
import torch
import torch.nn.functional as F
from torch import nn

from .inferencenumpts import make_fcos_postprocessor
# from .inference_reorg import make_fcos_postprocessor
from .loss_mutiPt import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class RFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(RFCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.num_pts = cfg.MODEL.FCOS.NUM_PTS

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, self.num_pts*num_classes, kernel_size=3, stride=1,
            padding=1
        )
        # x1y1 x2y2 h
        # self.bbox_pred = nn.Conv2d(
        #     in_channels, self.num_pts, kernel_size=3, stride=1,
        #     padding=1
        # )
        # x1y1 x2y2 h
        self.bbox_pred = nn.Conv2d(
            in_channels, self.num_pts*5, kernel_size=3, stride=1,
            padding=1
        )

        self.centerness = nn.Conv2d(
            in_channels, self.num_pts*1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            # 对FPN的结果进行卷积
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))

            # 每一层赋予一个权重
            # bbox_reg.append(torch.exp(self.scales[l](
            #     self.bbox_pred(self.bbox_tower(feature))
            # )))
            bbox_reg.append(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            ))
        return logits, bbox_reg, centerness

 
class RFCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RFCOSModule, self).__init__()

        head = RFCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        '''
        locations       不同分辨率 每个点中心xy坐标
        box_cls         logits
        box_regression  对logits使用torch.exp 
        '''
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        '''locations 长度为5'''
        locations = []
        # P3 - P7 
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        '''为不同分辨率卷积结果 都计算locations'''
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        # shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        # shift_x = shift_x.reshape(-1)
        # shift_y = shift_y.reshape(-1)
        # #                   ｎ -> n 2
        # shift_lt = torch.stack((shift_x, shift_y), dim=1) + stride // 4
        # shift_rt = torch.stack((shift_x + 3*stride // 4, shift_y+ stride // 4), dim=1)
        # shift_ct = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        # shift_ld = torch.stack((shift_x + stride // 4, shift_y+ 3*stride // 4), dim=1)
        # shift_rd = torch.stack((shift_x, shift_y), dim=1) + 3*stride // 4
        # n 2 -> 5*n 2
        # locations = torch.cat((shift_lt, shift_rt, shift_ct, shift_ld, shift_rd), dim=0)


        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        # ｎ -> n 2
        shift_x, shift_y = shift_x+stride//2, shift_y+stride//2
        # shift_x_1, shift_y_1 = shift_ct_x-2, shift_ct_y-2#2是相对中心坐标偏移量 增加多样性
        # shift_rt_x, shift_rt_y = shift_ct_x+2, shift_ct_y-2
        # shift_ld_x, shift_ld_y = shift_ct_x-2, shift_ct_y+2
        # shift_rd_x, shift_rd_y = shift_ct_x+2, shift_ct_y+2
        # n n 10 所在位置id%2 = 0则预测大目标
        locations = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=2)
        locations = locations.reshape(-1,2)

        return locations

def build_rfcos(cfg, in_channels):
    return RFCOSModule(cfg, in_channels)
