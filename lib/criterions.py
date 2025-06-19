import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class CLIP_loss(nn.Module):
    def __init__(self, weight=1.0):
        super(CLIP_loss, self).__init__()

        self.weight = weight
        print('=========using single label loss like CLIP==========')

    def forward(self, outputs, labels):

        labels = torch.arange(outputs[0].shape[0]).cuda()

        total_loss = (
            F.cross_entropy(outputs[0], labels) +
            F.cross_entropy(outputs[1], labels)
        ) / 2.0

        return self.weight*total_loss

class CLIP_multiloss(nn.Module):
    def __init__(self, weight=1.0):
        super(CLIP_multiloss, self).__init__()

        self.weight = weight

    def forward(self, outputs, labels):

        ones = torch.ones(labels.shape).cuda()
        vector = torch.hstack((labels, ones))
        cos1 = vector @ vector.T
        mod = torch.sqrt((vector * vector).sum(dim = 1,keepdim = True))
        cos2 = (mod @ mod.T)
        similar = cos1 / cos2
        multi_labels = similar.cpu().numpy()
        multi_labels = np.floor(multi_labels)
        # multi_labels = torch.Tensor(multi_labels).cuda().to(torch.long)
        multi_labels = torch.Tensor(multi_labels).cuda()

        # import ipdb
        # ipdb.set_trace()

        # total_loss = (
        #     F.binary_cross_entropy_with_logits(outputs[0], multi_labels) +
        #     F.binary_cross_entropy_with_logits(outputs[1], multi_labels)
        # ) / 2.0

        # total_loss = (
        #     F.kl_div(outputs[0], multi_labels) +
        #     F.kl_div(outputs[1], multi_labels)
        # ) / 2.0    

        total_loss = (
            F.smooth_l1_loss(outputs[0], multi_labels) +
            F.smooth_l1_loss(outputs[1], multi_labels)
        ) / 2.0             

        return self.weight*total_loss

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and support multi-label==========')
        self.error_metric = error_metric

    def forward(self, prediction, labels):

        # num = labels.shape[0]
        gt = torch.zeros((prediction[0].shape[0], prediction[0].shape[1])).cuda()
        for i, label in enumerate(labels):
            for k in range(labels.shape[0]):
                if labels[k] == label:
                    gt[i, k] = 1.0

        batch_size = prediction[0].shape[0]
        probs1 = F.log_softmax(prediction[0], 1) # B, C
        probs2 = F.log_softmax(prediction[1], 1) # C, B
        probs3 = F.softmax(gt * 10, 1) # B, C
        loss1 = self.error_metric(probs1, probs3) * batch_size
        loss2 = self.error_metric(probs2, probs3.t()) * batch_size
        losses = (loss1 + loss2)/ 2.0
        return losses

class MixLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric1=nn.KLDivLoss(size_average=True, reduce=True), 
                       error_metric2=nn.MSELoss().cuda()):
        super().__init__()
        print('=========using KL Loss=and MSE Loss together==========')
        self.error_metric1 = error_metric1
        self.error_metric2 = error_metric2

    def forward(self, prediction, labels):

        ones = (torch.ones(labels.shape)*0.5).cuda()
        vector = torch.hstack((labels, ones))
        v1 = vector[:,:,None].repeat(1,1,labels.shape[0])
        v2 = vector.transpose(0,1)[None,:,:].repeat(labels.shape[0],1,1)
        similarity = F.cosine_similarity(v1,v2)
        gt2 = torch.where(similarity < 1.0, 0.0, 1.0)

        pad = torch.zeros((prediction[0].shape[0], prediction[0].shape[1] - prediction[0].shape[0])).cuda()
        gt2 = torch.cat((gt2, pad), 1)

        batch_size = prediction[0].shape[0]
        probs1 = F.log_softmax(prediction[0], 1)
        probs2 = F.log_softmax(prediction[1], 1)
        probs3 = F.softmax(gt2 * 10, 1)
        loss1 = self.error_metric1(probs1, probs3) * batch_size
        loss2 = self.error_metric1(probs2, probs3.t()) * batch_size
        clip_loss = (loss1 + loss2)/ 2.0

        mse_loss = self.error_metric2(prediction[2], labels)

        sum_loss = clip_loss + mse_loss
        return sum_loss, clip_loss.data.cpu(), mse_loss.data.cpu()

class KLLoss_fast(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and support multi-label==========')
        self.error_metric = error_metric

    def forward(self, prediction, labels):

        # # original version
        # gt = torch.zeros((prediction[0].shape[0], prediction[0].shape[1])).cuda()
        # for i, label in enumerate(labels):
        #     for k in range(labels.shape[0]):
        #         if labels[k] == label:
        #             gt[i, k] = 1.0
        
        # # fast version
        ones = (torch.ones(labels.shape)*0.5).cuda()
        vector = torch.hstack((labels, ones))
        v1 = vector[:,:,None].repeat(1,1,labels.shape[0])
        v2 = vector.transpose(0,1)[None,:,:].repeat(labels.shape[0],1,1)
        similarity = F.cosine_similarity(v1,v2)
        gt2 = torch.where(similarity < 1.0, 0.0, 1.0)

        pad = torch.zeros((prediction[0].shape[0], prediction[0].shape[1] - prediction[0].shape[0])).cuda()
        gt2 = torch.cat((gt2, pad), 1)

        # check
        # print((gt2 == gt).all())

        batch_size = prediction[0].shape[0]
        probs1 = F.log_softmax(prediction[0], 1)
        probs2 = F.log_softmax(prediction[1], 1)
        probs3 = F.softmax(gt2 * 20, 1)
        loss1 = self.error_metric(probs1, probs3) * batch_size
        loss2 = self.error_metric(probs2, probs3.t()) * batch_size
        losses = (loss1 + loss2)/ 2.0
        
        return losses


class MixKDLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric1=nn.KLDivLoss(size_average=True, reduce=True),
                 error_metric2=nn.MSELoss().cuda()):
        super().__init__()
        print('=========using LLM based KD =and MSE Loss together==========')
        self.error_metric1 = error_metric1
        self.error_metric2 = error_metric2
        # add compute distance between specific prompt and corresponding output!

    def forward(self, prediction, labels):

        ones = (torch.ones(labels.shape)*0.5).cuda()
        vector = torch.hstack((labels, ones))
        v1 = vector[:,:,None].repeat(1,1,labels.shape[0])
        v2 = vector.transpose(0,1)[None,:,:].repeat(labels.shape[0],1,1)
        similarity = F.cosine_similarity(v1,v2)
        gt2 = torch.where(similarity < 1.0, 0.0, 1.0)

        pad = torch.zeros((prediction[0].shape[0], prediction[0].shape[1] - prediction[0].shape[0])).cuda()
        gt2 = torch.cat((gt2, pad), 1)

        mse_loss = self.error_metric2(prediction[2], labels)

        sum_loss = clip_loss + mse_loss
        return sum_loss, clip_loss.data.cpu(), mse_loss.data.cpu()