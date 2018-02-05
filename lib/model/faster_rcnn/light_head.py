import torch
import torch.nn as nn
import torch.nn.functional as F
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss
from torch.autograd import Variable

from .resnet import resnet101


class thin_features(nn.Module):
    def __init__(self):
        super(thin_features, self).__init__()

        self.left1 = nn.Conv2d(2048, 256, (15, 1), padding=(7, 0))
        self.left2 = nn.Conv2d(256, 512, (1, 15), padding=(0, 7))

        self.right1 = nn.Conv2d(2048, 256, (1, 15), padding=(0, 7))
        self.right2 = nn.Conv2d(256, 512, (15, 1), padding=(7, 0))

    def forward(self, x):
        x_left = F.relu(self.left1(x))
        x_left = F.relu(self.left2(x_left))

        x_right = F.relu(self.right1(x))
        x_right = F.relu(self.right2(x_right))

        return x_left + x_right


class LightHead(nn.Module):
    def __init__(self, classes, class_agnostic=True):
        super(LightHead, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.pretrained = True

        # define rpn
        self.RCNN_rpn = _RPN(1024)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / (16.0 * 2))

        self.grid_size = cfg.POOLING_SIZE

        self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
        resnet = resnet101()

        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict(
                {k: v for k, v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet.
        self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
        self.resnet4 = resnet.layer4

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.features.apply(set_bn_fix)
        self.resnet4.apply(set_bn_fix)

        self.thin_features = thin_features()

        self.fc1 = nn.Linear(512 * cfg.POOLING_SIZE * cfg.POOLING_SIZE, 512)
        self.bbox_pred = nn.Linear(512, 4)
        self.cls_pred = nn.Linear(512, self.n_classes)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.features(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        conv_feat = self.resnet4(base_feat)
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        conv_feat = self.thin_features(conv_feat)
        # TODO: downscale rois
        pooled_feat = self.RCNN_roi_pool(conv_feat, rois.view(-1, 5))
        pooled_feat = pooled_feat.view(-1, 512 * cfg.POOLING_SIZE * cfg.POOLING_SIZE)
        fc1 = self.fc1(pooled_feat)

        # compute bbox offset
        bbox_pred = self.bbox_pred(fc1)

        # compute object classification probability
        cls_score = self.cls_pred(fc1)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws,
                                             rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_weights()
