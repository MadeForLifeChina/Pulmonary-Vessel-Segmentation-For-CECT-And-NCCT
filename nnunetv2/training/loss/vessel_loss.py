import torch
from torch import nn
from nnunetv2.utilities.helpers import softmax_helper_dim1

def GetRegionWeigthFromRaio(weight,ratio):
    new_weight = torch.ones(weight.shape).cuda(weight.device.index)
    new_weight[weight>0.5] = ratio
    return new_weight
class WeightCrossEntropyLoss(nn.Module):
    def __init__(self, region_weight_ratio=4,class_weight_ratio=3,norm_weight=False):
        super(WeightCrossEntropyLoss, self).__init__()
        self.region_weight_ratio = region_weight_ratio
        self.class_weight_ratio = class_weight_ratio
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.norm_weight = norm_weight

    def forward(self, pred, target, region_weight=None):
        '''
        pred:  (batch,channel,x,y,z), channel = 3; 0--bg,1--artery,2--vein
        target:(batch,1,x,y,z), c=1 and the value should be 0,1,2,3,4  0--bg,1--artery,2--vein,3--a_centerline,4--v_centerline
        region_weight:(batch,c,x,y,z), voxel value denote weights of each voxel
        '''
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]

        pred_1 = pred.view(batch_size,num_classes,-1)
        targ_1 = target[:,0,:].view(batch_size,-1).long()

        loss = self.ce(pred_1,targ_1) # shape: (b,x*y*z)


        if region_weight is not None:
            region = region_weight[:, 0, :].view(batch_size,-1)
            #class_weight = GetRegionWeigthFromRaio(targ_1,self.class_weight_ratio)
            class_weight = targ_1.clone()
            class_weight[class_weight > 0.5] = 1
            class_weight = GetRegionWeigthFromRaio(class_weight, self.class_weight_ratio)
            cl_wight = GetRegionWeigthFromRaio(region,self.region_weight_ratio)
            weight = class_weight + cl_wight
        else:
            weight = torch.ones(targ_1.shape).cuda(targ_1.device.index)
        if self.norm_weight:
            weight = weight / weight.view(batch_size, -1).sum(axis=1).view([batch_size] + [1] * (len(weight.shape) - 1))

        loss = loss * weight
        loss = loss.mean()

        return loss

class WeightDiceLoss(nn.Module):
    def __init__(self, apply_nonlin = softmax_helper_dim1, batch_dice=True, do_bg=False, smooth= 1.,
                 region_weight_ratio=4,class_weight_ratio=3,norm_weight=False):
        super(WeightDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.region_weight_ratio = region_weight_ratio
        self.class_weight_ratio = class_weight_ratio
        self.norm_weight = norm_weight

    def forward(self, x, y,region_weight=None):
        # x : b,c,x,y,z
        # y : b,1,x,y,z
        # region_wight : b,1,x,y,z
        shp_x = x.shape
        if self.batch_dice:  # sum dice for images in each batch
            axes = [0] + list(range(2, len(shp_x)))
        else:  # calculate the average dice for images in each batch
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if region_weight is not None:
            #class_weight = GetRegionWeigthFromRaio(y, self.class_weight_ratio)
            class_weight = y.clone()
            class_weight[class_weight > 0.5] = 1
            class_weight = GetRegionWeigthFromRaio(class_weight, self.class_weight_ratio)
            cl_weight = GetRegionWeigthFromRaio(region_weight,self.region_weight_ratio)
            weight = class_weight + cl_weight

        else:
            weight = torch.ones(y.shape).cuda(y.device.index)
        if self.norm_weight:
            weight = weight / weight.view(shp_x[0], -1).sum(axis=1).view([shp_x[0]] + [1] * (len(weight.shape) - 1))

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, square=False, weight=weight)
        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class soft_cldice_loss(nn.Module):
    def __init__(self, apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, thresh_width=10,
                 kernel_size=3, padding=1, region_weight_ratio=4, class_weight_ratio=3, norm_weight=False):

        super(soft_cldice_loss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.thresh_width = thresh_width
        self.kernel_size = kernel_size
        self.padding = padding
        self.region_weight_ratio = region_weight_ratio
        self.class_weight_ratio = class_weight_ratio
        self.norm_weight = norm_weight

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
            if which_to_weight_class == 0:
                class_weight = clf.clone()
            else:
                class_weight = vf.clone()

            class_weight[class_weight > 0.5] = 1
            class_weight = GetRegionWeigthFromRaio(class_weight,self.class_weight_ratio)
            weight = GetRegionWeigthFromRaio(region_weight.view(*region_weight.shape[:2], -1),
                                             self.region_weight_ratio) + class_weight

        else:
            weight = torch.ones(clf.shape).cuda(clf.device.index)
        if self.norm_weight:
            weight = weight / weight.view(weight.shape[0], -1).sum(axis=1).view(
                [weight.shape[0]] + [1] * (len(weight.shape) - 1))

        intersection = (clf * vf * weight).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def soft_skeletonize_3D(self, x, thresh_width, kernel_size, padding):

        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool3d(x * -1, (kernel_size, kernel_size, kernel_size), 1,
                                                        padding) * -1
            contour = torch.nn.functional.relu(
                torch.nn.functional.max_pool3d(min_pool_x, (kernel_size, kernel_size, kernel_size), 1,
                                               padding) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def GetOneHotTarget(self, target, target_shape, device):
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

    def forward(self, net_output, target, region_weight=None):
        """
        net_output.shape = b,c,x,y,z
        targe.shape = b,1,x,y,z
        """
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        shp_x = net_output.shape
        shp_y = target.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))
            target_onehot = self.GetOneHotTarget(target, net_output.shape, net_output.device)

        if not self.do_bg:
            net_output = net_output[:, 1:].float()
            target_onehot = target_onehot[:, 1:].float()
        output_skeleton = self.soft_skeletonize_3D(net_output, thresh_width=self.thresh_width,
                                                   kernel_size=self.kernel_size,
                                                   padding=self.padding).float()
        target_skeleton = self.soft_skeletonize_3D(target_onehot, thresh_width=self.thresh_width,
                                                   kernel_size=self.kernel_size, padding=self.padding).float()

        iflat = self.__norm_intersection(output_skeleton, target_onehot, region_weight, which_to_weight_class=1)
        tflat = self.__norm_intersection(target_skeleton, net_output, region_weight, which_to_weight_class=0)
        intersection = iflat * tflat

        if self.batch_dice:
            intersection = intersection.sum(0)
            iflat = iflat.sum(0)
            tflat = tflat.sum(0)

        loss = -((2. * intersection) / (iflat + tflat)).mean()

        return loss

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
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if weight is not None:
        tp = tp * weight
        fp = fp * weight
        fn = fn * weight
        tn = tn * weight

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
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn

class clDC_DC_CE_clWeight_Loss(nn.Module):
    def __init__(self):
        super(clDC_DC_CE_clWeight_Loss,self).__init__()
        self.weight_dc = 1
        self.weight_ce = 1
        self.weight_cldc = 0.5

        self.ce = WeightCrossEntropyLoss()
        self.dc = WeightDiceLoss()
        self.cldc = soft_cldice_loss()

    def forward(self, x, y):
        # y shape: b, 1, x, y, z:
        # 0->bg, 1-->artery, 2-->vessel, 3-->artery_centerline, 4-->vein_centerline
        # x shape: b, 3, x, y, z
        # 0->bg, 1-->artery, 2-->vessel
        net_output = x
        target = y.clone()
        centerline = torch.zeros_like(target).cuda(y.device.index)
        centerline[target == 3] = 1
        centerline[target == 4] = 1
        target[target == 3] = 1 #add a_cl to artery label
        target[target == 4] = 2 #add v_cl to vein label
        dc_loss = self.dc(net_output, target, region_weight=centerline)
        ce_loss = self.ce(net_output, target, region_weight=centerline)
        cldc_loss = self.cldc(net_output, target, region_weight=centerline)
        loss = self.weight_ce * ce_loss + self.weight_dc * dc_loss + self.weight_cldc * cldc_loss


        return loss

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    x = torch.rand((2, 3, 168, 168, 168)).to(device)
    y = torch.randint(0, 5, (2, 1, 168, 168, 168)).to(device)

    loss_func = clDC_DC_CE_clWeight_Loss()

    import numpy as np


    loss = loss_func(x, y)
    print(loss)


