#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def GetRegionWeigthFromRaio(weight,ratio):
    '''
    weight shape: anything is ok
    '''
    new_weight = torch.ones(weight.shape).cuda(weight.device.index)
    new_weight[weight>0.5] = ratio
    # B = new_weight.shape[0]
    # norm_value = new_weight.view(B,-1).sum(axis=1).view([B]+[1]*(len(new_weight.shape)-1))
    # result = new_weight/norm_value 
    # # print('get weight',new_weight.shape,norm_value.shape,result.shape)
    # return new_weight/norm_value 
    return new_weight

class GDL(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, square_volumes=False):
        """
        square_volumes will square the weight term. The paper recommends square_volumes=True; I don't (just an intuition)
        """
        super(GDL, self).__init__()

        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(x.shape, y.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x)
            if x.device.type == "cuda":
                y_onehot = y_onehot.cuda(x.device.index)
            y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, self.square)

        # GDL weight computation, we use 1/V
        volumes = sum_tensor(y_onehot, axes) + 1e-6 # add some eps to prevent div by zero

        if self.square_volumes:
            volumes = volumes ** 2

        # apply weights
        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes

        # sum over classes
        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        # compute dice
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        dc = dc.mean()

        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False, weight=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    weight.shape=(b,1,x,y,z)
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1) # change GT to onehot [b,0,x,y,z] is bg, [b,1,x,y,z] is fg  
 
    # net_output.shape=(2,3,128,112,160), y_onehot.shape=(2,3,128,112,160)
    tp = net_output * y_onehot    
    fp = net_output * (1 - y_onehot)   
    fn = (1 - net_output) * y_onehot       
    tn = (1 - net_output) * (1 - y_onehot) 

    if weight is not None:
        tp = tp*weight
        fp = fp*weight
        fn = fn*weight
        tn = tn*weight

    # tp.fp.fn.tn.shape (batch_size,channel,x,y,z)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False) #c,1,1,1
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class RegionCrossEntropyLoss(nn.Module):
    def __init__(self, region_weight_ratio=1,class_weight_ratio=1,norm_weight=True):
        super(RegionCrossEntropyLoss, self).__init__()
        self.region_weight_ratio = region_weight_ratio
        self.class_weight_ratio = class_weight_ratio
        # self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([1]+self.class_weight_ratio).cuda(),reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.norm_weight = norm_weight
        # self.do_lung_mask = do_lung_mask

    def forward(self, pred, target, loss_mask=None, region_weight=None):
        '''
        pred:  (batch,channel,x,y,z), channel equals the number of class
        target:(batch,c,x,y,z), voxel value should be 0,1,2... denote different classes
        region_weight:(batch,c,x,y,z), voxel value denote weights of each voxel
        loss_mask: (batch,c,x,y,z), lung mask
        '''
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        # if (loss_mask is not None) & self.do_lung_mask:
        #     pred = pred*loss_mask
        #     target = target*loss_mask

        pred_1 = pred.view(batch_size,num_classes,-1)
        targ_1 = target[:,0,:].view(batch_size,-1).long()
        loss = self.ce(pred_1,targ_1) # shape: (b,x*y*z)

        if self.class_weight_ratio>1:
            class_weight = targ_1.clone().detach()
            # class_weight_1 = class_weight_1.cpu().numpy()
            # print(self.class_weight_ratio,self.region_weight_ratio,np.where(class_weight_1>0.5))
            class_weight[class_weight>0.5] = 1
            weight = class_weight*self.class_weight_ratio
            weight[weight==0] = 1
        else:
            weight = torch.ones(targ_1.shape).cuda(targ_1.device.index)


        if region_weight is not None and self.region_weight_ratio>1:
            weight = weight+GetRegionWeigthFromRaio(region_weight[:,0,:].view(batch_size,-1),self.region_weight_ratio)
            # print('ce loss',self.do_lung_mask,self.norm_weight,self.region_weight_ratio,self.class_weight_ratio,np.unique(weight.cpu()))
        
        if self.norm_weight:
            weight = weight / weight.view(batch_size,-1).sum(axis=1).view([batch_size]+[1]*(len(weight.shape)-1))
        
        loss = loss * weight
        loss = loss.mean()
        return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,\
                 region_weight_ratio=1,class_weight_ratio=1,norm_weight=False):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.region_weight_ratio = region_weight_ratio
        self.class_weight_ratio = class_weight_ratio
        self.norm_weight = norm_weight
        # self.do_lung_mask = do_lung_mask

    def forward(self, x, y, loss_mask=None, region_weight=None):
        '''
        x.shape (b,c,x,y,z)
        y.shape (b,1,x,y,z)
        region weight: shape, (batch,1,x,y,z)
        '''
        # print('--------------------soft dice: batch_dice',self.batch_dice,' do_bg',self.do_bg)
        # print('-----------------------x,y shape',x.shape,y.shape)
        shp_x = x.shape

        # self.batch_dice = True
        if self.batch_dice: # sum dice for images in each batch
            axes = [0] + list(range(2, len(shp_x)))
        else: # calculate the average dice for images in each batch
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)


        # with torch.no_grad():
        if region_weight is not None and self.region_weight_ratio>1:
            class_weight = y.clone().detach()[:,:1,:]
            class_weight[class_weight>0.5] = 1
            weight = GetRegionWeigthFromRaio(region_weight,self.region_weight_ratio) + class_weight*self.class_weight_ratio
            # print('dice loss',self.do_lung_mask,self.norm_weight,self.region_weight_ratio,self.class_weight_ratio,np.unique(weight.cpu()))
        else:
            weight = None

        if self.class_weight_ratio>1:
            class_weight = y.clone()[:,:1,:].detach()
            class_weight[class_weight>0.5] = 1
            if weight == None:
                weight = class_weight*self.class_weight_ratio
                weight[weight==0] = 1
            else:
                weight1 = class_weight*self.class_weight_ratio
                weight1[weight1==0] = 1                
                weight += weight1
         
        if self.norm_weight:
            weight = weight / weight.view(shp_x[0],-1).sum(axis=1).view([shp_x[0]]+[1]*(len(weight.shape)-1))
    

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False, weight=weight)
        # tp,fp,fn.shape: (channel)

        nominator = 2 * tp  + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean() # mean for all classes

        return -dc


class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

# self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result




class soft_cldice_loss(nn.Module):
    def __init__(self, output_folder, apply_nonlin=None, batch_dice=True, do_bg=False, thresh_width=12,\
                kernel_size=5, padding=2, region_weight_ratio=1,class_weight_ratio=1,norm_weight=True,do_lung_mask=True): #aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 # log_dice=False, ignore_label=None):
        """
        
        """
        super(soft_cldice_loss, self).__init__()
        self.output_folder = output_folder
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.thresh_width = thresh_width
        self.kernel_size = kernel_size
        self.padding = padding
        self.region_weight_ratio = region_weight_ratio
        self.class_weight_ratio = class_weight_ratio
        self.norm_weight = norm_weight
        self.do_lung_mask = do_lung_mask
        
    def __norm_intersection(self, center_line, vessel, region_weight=None, which_to_weight_class=0):
        '''
        inputs shape  (batch, channel, height, width, z)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        region_weight shape: 2,channel, h, w, z
        '''

        
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)

        if region_weight is not None:
            if which_to_weight_class==0:
                class_weight = clf.clone()
            else:
                class_weight = vf.clone()
            # class_weight = class_weight.sum(axis=1)[:,None]
            class_weight[class_weight>0.5] = 1
            weight = GetRegionWeigthFromRaio(region_weight.view(*region_weight.shape[:2],-1),self.region_weight_ratio)+\
                        class_weight*self.class_weight_ratio
            # print('cldice',self.region_weight_ratio,self.class_weight_ratio,np.unique(weight.cpu()),np.unique(class_weight.cpu()))
        else:
            weight = torch.ones(clf.shape).cuda(clf.device.index)
        if self.norm_weight:
            weight = weight / weight.view(weight.shape[0],-1).sum(axis=1).view([weight.shape[0]]+[1]*(len(weight.shape)-1))
        # print('cldice-----------',vessel.shape,vf.shape,clf.shape,weight.shape,which_to_weight_class,region_weight.shape,class_weight.shape)
        intersection = (clf * vf * weight).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def __soft_skeletonize_2D(self, x, thresh_width=5, kernel_size=3, padding=1):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        x: (minibatch,in_channel,iH,iW)
        '''
        # x=torch.nn.functional.max_pool2d(x, (kernel_size, kernel_size), 1, padding)
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x*-1, (kernel_size, kernel_size), 1, padding)*-1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (kernel_size, kernel_size), 1, padding) - min_pool_x)
            
            x = torch.nn.functional.relu(x - contour)
        return x

    def soft_skeletonize_3D(self, x, thresh_width=12, kernel_size=5, padding=2): #12,5,2
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        x: (minibatch,in_channel,iT,iH,iW)
        '''
        # x=torch.nn.functional.max_pool3d(x, (kernel_size, kernel_size, kernel_size), 1, padding)
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool3d(x*-1, (kernel_size, kernel_size, kernel_size), 1, padding)*-1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x,(kernel_size, kernel_size, kernel_size), 1, padding) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def GetMIP(self, data, axis=2):
        # print(data.shape)
        # print(np.max(data,axis=2))
        return torch.max(data,axis=axis)[0].cpu().data.numpy()

    def GetOneHotTarget(self,target,target_shape,device):
        if all([i == j for i, j in zip(target_shape, target.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            target_onehot = target
        else:
            gt = target.long()
            target_onehot = torch.zeros(target_shape)
            if device.type == "cuda":
                target_onehot = target_onehot.cuda(device.index)
            target_onehot.scatter_(1, gt, 1)
        return target_onehot

    def forward(self, net_output, target, lung_mask=None, region_weight=None):
        """
        net_output.shape = 2,2,128,112,160
        targe.shape = 2,1,128,112,160
        
        target must be b, c, x, y(, z) with c=1 
        :param net_output: (b, c, x, y(, z)))
        :param target: (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
        :return:
        """
        # print('cldice loss',self.do_lung_mask,self.norm_weight,np.unique(target.cpu()))
        if (lung_mask is not None) & self.do_lung_mask:
            target = target*lung_mask
            net_output = net_output*lung_mask
            # print('cldice loss',self.do_lung_mask,self.norm_weight,np.unique(target.cpu()))
        # print('----------------------cldice',np.unique(target.cpu()),np.unique(net_output.detach().cpu()))
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
            target = self.apply_nonlin(target) # commit for 2.6 12:30 for cl5
        # print('----------------------cldice',net_output.shape,target.shape)
        # print('----------------------cldice2',np.unique(target.cpu()),np.unique(net_output.detach().cpu()))

        shp_x = net_output.shape
        shp_y = target.shape
        # print(shp_x,shp_y,'-------------')
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))
            # print(target.shape,np.unique(target.cpu()))
            target_onehot = self.GetOneHotTarget(target,net_output.shape,net_output.device)
            # print(np.unique(target.cpu()),np.unique(target_onehot.cpu()))
            # if all([i == j for i, j in zip(net_output.shape, target.shape)]):
            #     # if this is the case then gt is probably already a one hot encoding
            #     target_onehot = target
            # else:
            #     target = target.long()
            #     target_onehot = torch.zeros(shp_x)
            #     if net_output.device.type == "cuda":
            #         # print(net_output.device.index,target_onehot.shape,np.unique(target_onehot))
            #         target_onehot = target_onehot.cuda(net_output.device.index)
            #     target_onehot.scatter_(1, target, 1) # change GT to onehot [b,0,x,y,z] is bg, [b,1,x,y,z] is fg  

        if not self.do_bg:
            net_output = net_output[:,1:].float()
            target_onehot = target_onehot[:,1:].float()
        output_skeleton = self.soft_skeletonize_3D(net_output,thresh_width=self.thresh_width, kernel_size=self.kernel_size, padding=self.padding).float() #.long()
        target_skeleton = self.soft_skeletonize_3D(target_onehot,thresh_width=self.thresh_width, kernel_size=self.kernel_size, padding=self.padding).float()
        # print('++++++++++++++++++++++++++++++++++++',net_output.shape,target.shape,output_skeleton.shape,target_skeleton.shape)

        iflat = self.__norm_intersection(output_skeleton, target_onehot, region_weight, which_to_weight_class=1)
        tflat = self.__norm_intersection(target_skeleton, net_output, region_weight, which_to_weight_class=0)
        intersection = iflat * tflat
        # print('intersection shape',intersection.shape)

        if self.batch_dice:
            intersection = intersection.sum(0)
            iflat = iflat.sum(0)
            tflat = tflat.sum(0)

        result = -((2. * intersection)/(iflat + tflat)).mean()

        # batch_s = net_output.shape[0]
        # dim_z = net_output.shape[4]

        # fig= plt.figure(figsize=(12,6))
        # for b in range(batch_s):
        #     # target
        #     ax = plt.subplot2grid((batch_s,4), (b,0))
        #     ax.imshow(self.__GetMIP(target[b,0,:,:,dim_z-5:dim_z+5]),cmap=plt.cm.bone)
        #     # target skeleton
        #     ax = plt.subplot2grid((batch_s,4), (b,1))
        #     ax.imshow(self.__GetMIP(target_skeleton[b,0,:,:,dim_z-5:dim_z+5]),cmap=plt.cm.bone)
        #     # prediction
        #     ax = plt.subplot2grid((batch_s,4), (b,2))
        #     ax.imshow(self.__GetMIP(net_output[b,0,:,:,dim_z-5:dim_z+5]),cmap=plt.cm.bone)
        #     # prediction target
        #     ax = plt.subplot2grid((batch_s,4), (b,3))
        #     ax.imshow(self.__GetMIP(output_skeleton[b,0,:,:,dim_z-5:dim_z+5]),cmap=plt.cm.bone)
        # plt.subplots_adjust(left = 0.08,right = 0.95,bottom = 0.08,top = 0.95)
        # fig.savefig(self.output_folder+'/fold_1/skeleton.png')
        # plt.close()
        return result


class soft_cldice_and_DC_and_CE_loss(nn.Module):
    def __init__(self, output_folder, soft_dice_kwargs, ce_kwargs, cldice_kwargs,weight_kwargs,
                aggregate="sum", square_dice=False, log_dice=False, ignore_label=None,
                lung_mask_CE=False, lung_mask_dice=False):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(soft_cldice_and_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.weight_dice = weight_kwargs['weight_dice']
        self.weight_ce = weight_kwargs['weight_ce']
        self.weight_cldice = weight_kwargs['weight_cldice']
        self.log_dice = log_dice
        # self.weight_dice = weight_dice
        # self.weight_ce = weight_ce
        # self.weight_cldice = weight_cldice
        self.aggregate = aggregate
        self.lung_mask_CE = lung_mask_CE
        self.lung_mask_dice = lung_mask_dice
        # self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        # print('------------------------',ce_kwargs['c_weight_vessel'])
        if len(ce_kwargs['c_weight_vessel']) == 1:
            self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,ce_kwargs['c_weight_vessel'][0]]).cuda())
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,ce_kwargs['c_weight_vessel'][0],ce_kwargs['c_weight_vessel'][1]]).cuda())
        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

        self.cldc = soft_cldice_loss(output_folder,apply_nonlin=softmax_helper, **cldice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output: b,c,x,y,z
        :param target:
        :return:
        """
        # print('------------------------network output',net_output.shape)
        if target.shape[1] != 1:
            with_lung_mask = True
            lung_mask = target[:,:1,:].clone().detach()
            target = target[:,1:,:]
        else:
            with_lung_mask = False
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        if self.lung_mask_dice:
            dc_loss = self.dc(net_output*lung_mask, target*lung_mask, loss_mask=mask) if self.weight_dice != 0 else 0
        else:
            dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)


        if self.lung_mask_CE:
            ce_loss = self.ce(net_output*lung_mask, (target*lung_mask)[:, 0].long()) if self.weight_ce != 0 else 0
        else:
            ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0

        # print('dice loss',net_output.shape,lung_mask.shape,ce_loss.shape,ce_loss)
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.weight_cldice != 0:
            if with_lung_mask:
                cldc_loss = self.cldc(net_output, target, lung_mask)
            else:
                cldc_loss = self.cldc(net_output, target)
        else:
            cldc_loss = 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

def GetSemanticFromInstanceGT(ins_gt):
    result = ins_gt.clone()
    result[result>1] = 1
    return result

class soft_two_scale_loss(nn.Module):
    def __init__(self, output_folder, se_in_weight, soft_dice_kwargs, ce_kwargs, 
                cldice_kwargs,weight_kwargs,
                aggregate="sum", square_dice=False,log_dice=False, ignore_label=None, 
                lung_mask_CE=False, lung_mask_dice=False,lung_mask_cldice=False,
                fg_only_for_3rdNN=False,norm_weight=False,with_cl=False):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(soft_two_scale_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'

        self.mask_w, self.ins_l_w, self.ini_ins_s_w = se_in_weight
        self.ins_s_w = self.ini_ins_s_w

        if with_cl:
            self.ins_loss = soft_cldice_DC_CE_centerline_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs,weight_kwargs,\
                                                        norm_weight=norm_weight,lung_mask_dice=lung_mask_dice,\
                                                        lung_mask_CE=lung_mask_CE,lung_mask_cldice=lung_mask_cldice,\
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)

           
            self.sem_loss = soft_cldice_DC_CE_centerline_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs,weight_kwargs,\
                                                        norm_weight=norm_weight,lung_mask_dice=lung_mask_dice,\
                                                        lung_mask_CE=lung_mask_CE,lung_mask_cldice=lung_mask_cldice,\
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)

            if fg_only_for_3rdNN:
                soft_dice_kwargs['class_weight_ratio'] = 1000
                ce_kwargs['c_weight_vessel'] = 1000
                # print('------------weight',soft_dice_kwargs,ce_kwargs)
                self.s_ins_loss = soft_cldice_DC_CE_centerline_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs,weight_kwargs,\
                                                        norm_weight=norm_weight,lung_mask_dice=lung_mask_dice,\
                                                        lung_mask_CE=lung_mask_CE,lung_mask_cldice=lung_mask_cldice,\
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)
            else:
                self.s_ins_loss = soft_cldice_DC_CE_centerline_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs,weight_kwargs,\
                                                        norm_weight=norm_weight,lung_mask_dice=lung_mask_dice,\
                                                        lung_mask_CE=lung_mask_CE,lung_mask_cldice=lung_mask_cldice,\
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)
        else:
        
            self.sem_loss = soft_cldice_and_DC_and_CE_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs=cldice_kwargs, 
                                                        weight_kwargs=weight_kwargs,lung_mask_CE=lung_mask_CE, lung_mask_dice=lung_mask_dice,
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)

            ce_kwargs['c_weight_vessel'] = [ce_kwargs['c_weight_vessel'][0],ce_kwargs['c_weight_vessel'][0]]
            self.ins_loss = soft_cldice_and_DC_and_CE_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs=cldice_kwargs, 
                                                        weight_kwargs=weight_kwargs,lung_mask_CE=lung_mask_CE, lung_mask_dice=lung_mask_dice,
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)

            if fg_only_for_3rdNN:
                soft_dice_kwargs['class_weight_ratio'] = 1000
                ce_kwargs['c_weight_vessel'] = [1000,1000]
                # print('------------weight',soft_dice_kwargs,ce_kwargs)
                self.s_ins_loss = soft_cldice_and_DC_and_CE_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs=cldice_kwargs, 
                                                        weight_kwargs=weight_kwargs,lung_mask_CE=lung_mask_CE, lung_mask_dice=lung_mask_dice,
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)

            else:
                self.s_ins_loss = soft_cldice_and_DC_and_CE_loss(output_folder,soft_dice_kwargs,ce_kwargs, cldice_kwargs=cldice_kwargs, 
                                                        weight_kwargs=weight_kwargs,lung_mask_CE=lung_mask_CE, lung_mask_dice=lung_mask_dice,
                                                        aggregate=aggregate, square_dice=square_dice,log_dice=log_dice, ignore_label=ignore_label)

        
    def SetSInstanceLossWeight(self,value):
        self.ins_s_w = value

    def SetLInstanceLossWeight(self,value):
        self.ins_l_w = value

    def SetSSemanticLossWeight(self,value):
        self.mask_w = value

    def forward(self, net_output, target):
        """
        target must be [b, c, x, y(, z) with c=1,b, c, x, y(, z) with c=1]
        net_output = [snet_output,lnet_output]
        :param net_output:
        :param target:
        :return:
        """
        # print('------------------------network output',net_output.shape)
        # print('loss calculating')
        semantic_loss = self.sem_loss(net_output[0], GetSemanticFromInstanceGT(target[0]))
        instance_l_loss = self.ins_loss(net_output[1], target[1])
        instance_s_loss = self.s_ins_loss(net_output[2], target[0]) if self.ins_s_w != 0 else torch.tensor([0])[0].cuda(net_output[0].device.index)
        # print('instance loss',instance_s_loss) # -0.8~-0.7 for weight 1000
        result = self.mask_w*semantic_loss + self.ins_l_w*instance_l_loss + self.ins_s_w*instance_s_loss

        return result,semantic_loss,instance_l_loss,instance_s_loss

class soft_cldice_DC_CE_centerline_loss(nn.Module):
    def __init__(self, output_folder, soft_dice_kwargs, ce_kwargs, cldice_kwargs, weight_kwargs, aggregate="sum", square_dice=False,\
        log_dice=False, ignore_label=None, norm_weight=True,lung_mask_dice=False,lung_mask_CE=False,lung_mask_cldice=False):
        super(soft_cldice_DC_CE_centerline_loss,self).__init__()
        self.log_dice = log_dice
        self.weight_dice = weight_kwargs['weight_dice']
        self.weight_ce = weight_kwargs['weight_ce']
        self.weight_cldice = weight_kwargs['weight_cldice']
        self.aggregate = aggregate
        self.ignore_label = ignore_label
        self.lung_mask_dice = lung_mask_dice
        self.lung_mask_CE = lung_mask_CE
        self.lung_mask_cldice = lung_mask_cldice

        self.ce = RegionCrossEntropyLoss(**ce_kwargs,norm_weight=norm_weight)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs,norm_weight=norm_weight)
        self.cldc = soft_cldice_loss(output_folder,apply_nonlin=softmax_helper,**cldice_kwargs,norm_weight=norm_weight)


        
    def forward(self, net_output, target):
        # target shape: b, 3, x, y, z: 3 is 0 lung mask, 1 centerline, 2 vessel domain


        # pred:  (batch,channel,x,y,z), channel equals the number of class
        # target:(batch,x,y,z)
        # region_weight:(batch,x,y,z), voxel value denote weights of each voxel
        # loss_mask: (batch,x,y,z), lung mask
        # print('-------net_output shape',net_output.shape, 'target shape', target.shape)
        # print('loss',net_output.shape,target.shape)
        if target.shape[1] != 1:
            lung_mask = target[:,:1,:].clone().detach()
            centerline = target[:,1:2,:].clone().detach()
            target = target[:,-1:,:]
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        if self.lung_mask_dice:
            dc_loss = self.dc(net_output*lung_mask, target*lung_mask, loss_mask=mask, region_weight=centerline) if self.weight_dice != 0 else 0
        else:
            dc_loss = self.dc(net_output, target, loss_mask=mask, region_weight=centerline) if self.weight_dice != 0 else 0

        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        if self.lung_mask_CE:
            ce_loss = self.ce(net_output*lung_mask, target*lung_mask,region_weight=centerline) if self.weight_ce != 0 else 0
        else:
            ce_loss = self.ce(net_output, target,region_weight=centerline) if self.weight_ce != 0 else 0

        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.lung_mask_cldice:
            cldc_loss = self.cldc(net_output, target, lung_mask, region_weight=centerline) if self.weight_cldice != 0 else 0
        else:
            cldc_loss = self.cldc(net_output, target, region_weight=centerline) if self.weight_cldice != 0 else 0

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldc_loss
            # print('ce',ce_loss,', dc',dc_loss,', cldc',cldc_loss,',res',result)
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result



class soft_cldice_and_DC_and_CE_multi_task_centerline_loss(nn.Module):
    def __init__(self, output_folder, soft_dice_kwargs, ce_kwargs, cldice_kwargs, weight_kwargs, aggregate="sum", square_dice=False,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(soft_cldice_and_DC_and_CE_multi_task_centerline_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_kwargs['weight_dice']
        self.weight_ce = weight_kwargs['weight_ce']
        self.weight_cldice = weight_kwargs['weight_cldice']
        self.aggregate = aggregate
        self.ce = RegionCrossEntropyLoss(**ce_kwargs)
        # self.ce_2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,bce_kwargs['c_weight_vessel']]).cuda())
        # self.ce_3 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,bce_kwargs['c_weight_av'],bce_kwargs['c_weight_av']]).cuda())
        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

        self.cldc = soft_cldice_loss(output_folder,apply_nonlin=softmax_helper,**cldice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if target.shape[1] != 1:
            with_lung_mask = True
            lung_mask = target[:,:1,:]
            if target.shape[1] == 3:
                centerline = target[:,1:2,:]
            # print('unique',np.unique(lung_mask.cpu()),np.unique(centerline.cpu()))
            instance_target = target[:,-1:,:]
            target = instance_target.clone()
            target[instance_target>0.5] = 1
            # artery_target = instance_target.clone()
           # artery_target[instance_target==1] =1
            # print('target',target.shape,instance_target.shape)
        else:
            with_lung_mask = False

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None
        # print('output1',net_output.shape)
        semantic_output = net_output[:,:2,:].cuda(net_output.device.index)
        instance_output = net_output[:,2:,:].cuda(net_output.device.index)
        # print('output',semantic_output.shape,artery_output.shape)
        dc_loss_se = self.dc(semantic_output, target, region_weight=centerline) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss_se = -torch.log(-dc_loss_se)

        dc_loss_in = self.dc(instance_output, instance_target, region_weight=centerline) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss_in = -torch.log(-dc_loss_in)

        # target_onehot = self.cldc.GetOneHotTarget(target,semantic_output.shape,semantic_output.device)
        # print('shape',semantic_output.shape,target.shape)

        ce_loss = self.ce(semantic_output, target, region_weight=centerline) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        # artery_target_onehot = self.cldc.GetOneHotTarget(artery_target,artery_output.shape,artery_output.device)
        ce_loss_in = self.ce(instance_output, instance_target,region_weight=centerline) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss_in *= mask[:, 0]
            ce_loss_in = ce_loss_in.sum() / mask.sum()

        # print('-----------------shape,',semantic_output.shape,instance_output.shape,target.shape,instance_target.shape)
        if with_lung_mask:
            cldc_loss = self.cldc(semantic_output, target, lung_mask, region_weight=centerline)
        else:
            cldc_loss = self.cldc(semantic_output, target, region_weight=centerline)

        if with_lung_mask:
            cldc_loss_in = self.cldc(instance_output, instance_target, lung_mask, region_weight=centerline)
        else:
            cldc_loss_in = self.cldc(instance_output, instance_target, region_weight=centerline)

        if self.aggregate == "sum":
            result = self.weight_ce * (ce_loss+ce_loss_in) + self.weight_dice * (dc_loss_in+dc_loss_se) + self.weight_cldice * (cldc_loss+cldc_loss_in)
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result




class soft_cldice_and_DC_and_CE_multi_task_loss(nn.Module):
    def __init__(self, output_folder, soft_dice_kwargs, bce_kwargs, cldice_kwargs, weight_kwargs, aggregate="sum", square_dice=False,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(soft_cldice_and_DC_and_CE_multi_task_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_kwargs['weight_dice']
        self.weight_ce = weight_kwargs['weight_ce']
        self.weight_cldice = weight_kwargs['weight_cldice']
        self.aggregate = aggregate
        self.ce_2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,bce_kwargs['c_weight_vessel']]).cuda())
        self.ce_3 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,bce_kwargs['c_weight_av'],bce_kwargs['c_weight_av']]).cuda())
        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

        self.cldc = soft_cldice_loss(output_folder,apply_nonlin=softmax_helper,**cldice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if target.shape[1] != 1:
            with_lung_mask = True
            lung_mask = target[:,:1,:]
            instance_target = target[:,1:,:]
            target = instance_target.clone()
            target[instance_target>0.5] = 1
            artery_target = instance_target.clone()
           # artery_target[instance_target==1] =1
            # print('target',target.shape,instance_target.shape)
        else:
            with_lung_mask = False

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None
        # print('output1',net_output.shape)
        semantic_output = net_output[:,:2,:].cuda(net_output.device.index)
        artery_output = net_output[:,2:,:].cuda(net_output.device.index)
        # print('output',semantic_output.shape,artery_output.shape)
        dc_loss_se = self.dc(semantic_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss_se = -torch.log(-dc_loss_se)

        dc_loss_ar = self.dc(artery_output, artery_target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss_ar = -torch.log(-dc_loss_ar)

        # target_onehot = self.cldc.GetOneHotTarget(target,semantic_output.shape,semantic_output.device)
        # print('shape',semantic_output.shape,target.shape)

        ce_loss = self.ce_2(semantic_output, target[:,0,:].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        # artery_target_onehot = self.cldc.GetOneHotTarget(artery_target,artery_output.shape,artery_output.device)
        ce_loss_ar = self.ce_3(artery_output, artery_target[:,0,:].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss_ar *= mask[:, 0]
            ce_loss_ar = ce_loss_ar.sum() / mask.sum()

        if with_lung_mask:
            cldc_loss = self.cldc(semantic_output, target, lung_mask)
        else:
            cldc_loss = self.cldc(semantic_output, target)

        if with_lung_mask:
            cldc_loss_ar = self.cldc(artery_output, artery_target, lung_mask)
        else:
            cldc_loss_ar = self.cldc(artery_output, artery_target)

        if self.aggregate == "sum":
            result = self.weight_ce * (ce_loss+ce_loss_ar) + self.weight_dice * (dc_loss_ar+dc_loss_se) + self.weight_cldice * (cldc_loss+cldc_loss_ar)
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, aggregate="sum"):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.aggregate = aggregate
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(net_output, target)

        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result


class GDL_and_CE_loss(nn.Module):
    def __init__(self, gdl_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(GDL_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = GDL(softmax_helper, **gdl_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result
