import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU


class PAN_PP_V2(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(PAN_PP_V2, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)

        self.fpn = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.rec_head = None
        if recognition_head:
            self.rec_head = build_head(recognition_head)

        self.conv1x1 = nn.Conv2d(128*7, 128*4, kernel_size=1, stride=1, padding=0, bias=False)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        F1, M1, F2, M2, F3, M3, F4 = self.fpn(f1, f2, f3, f4)

        # FFM
        M1 = self._upsample(M1, F1.size())
        F2 = self._upsample(F2, F1.size())
        M2 = self._upsample(M2, F1.size())
        F3 = self._upsample(F3, F1.size())
        M3 = self._upsample(M3, F1.size())
        F4 = self._upsample(F4, F1.size())

        f = torch.cat((F1, M1, F2, M2, F3, M3, F4), 1)

        f = self.conv1x1(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))
            start = time.time()

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            loss_det = self.det_head.loss(det_out, gt_texts, gt_kernels,
                                          training_masks, gt_instances,
                                          gt_bboxes)
            outputs.update(loss_det)
        else:
            det_out = self._upsample(det_out, imgs.size(), cfg.test_cfg.scale)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        if self.rec_head is not None:
            if self.training:
                if cfg.train_cfg.use_ex:
                    x_crops, gt_words = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * gt_kernels[:, 0] * training_masks,
                        gt_bboxes, gt_words, word_masks)
                else:
                    x_crops, gt_words = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * training_masks, gt_bboxes, gt_words,
                        word_masks)

                if x_crops is not None:
                    out_rec = self.rec_head(x_crops, gt_words)
                    loss_rec = self.rec_head.loss(out_rec,
                                                  gt_words,
                                                  reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1, ), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1, ), -1, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
            else:
                if len(det_res['bboxes']) > 0:
                    x_crops, _ = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(det_res['label'],
                                     dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(det_res['bboxes_h'],
                                            dtype=torch.long),
                        unique_labels=det_res['instances'])
                    words, word_scores = self.rec_head.forward(x_crops)
                else:
                    words = []
                    word_scores = []

                if cfg.report_speed:
                    torch.cuda.synchronize()
                    outputs.update(dict(rec_time=time.time() - start))
                outputs.update(
                    dict(words=words, word_scores=word_scores, label=''))

        return outputs
