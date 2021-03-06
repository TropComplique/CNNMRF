import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


# a small value
EPSILON = 1e-8


class MarkovRandomFieldLoss(nn.Module):

    def __init__(self, y, size, stride, threshold):
        """
        Arguments:
            y: a list of float tensors with shape [1, c, a_i, b_i],
                where (a_i, b_i) is the spatial size of the i-th tensor.
            size, stride: integers, parameters of used patches.
            threshold: a float number.
        """
        super(MarkovRandomFieldLoss, self).__init__()

        y = [extract_patches(t.squeeze(0), size, stride) for t in y]
        y = torch.cat(y, dim=0)  # shape [N, c * size * size]
        y_normed = y/(y.norm(p=2, dim=1, keepdim=True) + EPSILON)

        self.y = nn.Parameter(data=y, requires_grad=False)
        self.y_normed = nn.Parameter(data=y_normed, requires_grad=False)

        self.size = size
        self.stride = stride
        self.threshold = threshold

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [1, c, h, w].
        Returns:
            a float tensor with shape [].
        """

        x = extract_patches(x.squeeze(0), self.size, self.stride)
        x_normed = x/(x.norm(p=2, dim=1, keepdim=True) + EPSILON)
        # they have shape [M, c * size * size]

        # normalized cross-correlations (cosine similarity)
        products = torch.matmul(self.y_normed, x_normed.t())  # shape [N, M]

        # maximal similarity
        values, indices = torch.max(products, dim=0)
        # they have shape [M]

        similar_enough = values.ge(self.threshold)
        indices = torch.masked_select(indices, similar_enough)
        # it has shape [M'], where M' <= M

        y = torch.index_select(self.y, 0, indices)  # shape [M', c * size * size]
        valid_indices = torch.nonzero(similar_enough).squeeze(1)  # shape [M']
        x = torch.index_select(x, 0, valid_indices)  # shape [M', c * size * size]

        return torch.pow(x - y, 2).mean([0, 1])  # shape []


def extract_patches(features, size, stride):
    """
    Arguments:
        features: a float tensor with shape [c, h, w].
        size: an integer, size of the patch.
        stride: an integer.
    Returns:
        a float tensor with shape [N, c * size * size],
        where N = n * m, n = 1 + floor((h - size)/stride),
        and m = 1 + floor((w - size)/stride).
    """
    c, h, w = features.size()
    patches = features.unfold(1, size, stride).unfold(2, size, stride)
    # it has shape [c, n, m, size, size]

    # get the number of patches
    n, m = patches.size()[1:3]
    N = n * m

    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(N, c * size * size)
    return patches


class PerceptualLoss(nn.Module):

    def __init__(self, f):
        super(PerceptualLoss, self).__init__()
        self.f = nn.Parameter(data=f, requires_grad=False)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, c, h, w].
        Returns:
            a float tensor with shape [].
        """
        return F.mse_loss(x, self.f, reduction='mean')


class TotalVariationLoss(nn.Module):

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """
        h, w = x.size()[2:]
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :(h - 1), :], 2)
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :(w - 1)], 2)
        return h_tv.mean([0, 1, 2, 3]) + w_tv.mean([0, 1, 2, 3])


class Extractor(nn.Module):

    def __init__(self, layers):
        """
        Arguments:
            layers: a list of strings.
        """
        super(Extractor, self).__init__()

        self.model = vgg19(pretrained=True).eval().features
        for p in self.model.parameters():
            p.requires_grad = False

        # normalization
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.mean = nn.Parameter(data=mean, requires_grad=False)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.std = nn.Parameter(data=std, requires_grad=False)

        names = []
        i, j = 1, 1
        for m in self.model:

            if isinstance(m, nn.Conv2d):
                names.append(f'conv{i}_{j}')

            elif isinstance(m, nn.ReLU):
                names.append(f'relu{i}_{j}')
                m.inplace = False
                j += 1

            elif isinstance(m, nn.MaxPool2d):
                names.append(f'pool{i}')
                i += 1
                j = 1

        # names of all features
        self.names = names

        # names of features to extract
        self.layers = list(set(layers))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents RGB images with pixel values in [0, 1] range.
        Returns:
            a dict with float tensors.
        """
        features = {}
        x = (x - self.mean)/self.std

        i = 0  # number of features extracted
        num_features = len(self.layers)

        for n, m in zip(self.names, self.model):
            x = m(x)

            if n in self.layers:
                features[n] = x
                i += 1

            if i == num_features:
                break

        return features
