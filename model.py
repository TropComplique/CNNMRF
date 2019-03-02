import torch
import torch.nn as nn
import numpy as np
from losses import PerceptualLoss, TotalVariationLoss, MarkovRandomFieldLoss, Extractor


# names of features to use
CONTENT_LAYERS = ['relu5_1']
STYLE_LAYERS = ['relu3_1', 'relu4_1', 'relu5_1']


class Loss(nn.Module):

    def __init__(self, content, styles, initial=None):
        """
        Arguments:
            content: an instance of PIL image.
            styles: a list of PIL images.
            initial: an instance of PIL image or None.
        """
        super(Loss, self).__init__()

        # image to start optimization from
        if initial is None:
            mean, std = 0.5, 1e-3
            w, h = content.size
            initial = mean + std * torch.randn(1, 3, h, w)
        else:
            assert initial.size == content.size
            initial = to_tensor(initial)

        # images
        content = to_tensor(content)
        styles = [to_tensor(s) for s in styles]
        self.x = nn.Parameter(data=initial, requires_grad=True)

        # features
        feature_names = CONTENT_LAYERS + STYLE_LAYERS
        self.vgg = Extractor(feature_names)
        cf = self.vgg(content)
        sf = [self.vgg(s) for s in styles]

        # create losses
        self.content = nn.ModuleDict({
            n: PerceptualLoss(cf[n])
            for n in CONTENT_LAYERS
        })
        self.style = nn.ModuleDict({
            n: MarkovRandomFieldLoss(
                [sf[i][n] for i in range(len(styles))],
                size=5, stride=2, threshold=1e-3
            )
            for n in STYLE_LAYERS
        })
        self.tv = TotalVariationLoss()

    def forward(self):

        f = self.vgg(self.x)
        content_loss = torch.tensor(0.0, device=self.x.device)
        style_loss = torch.tensor(0.0, device=self.x.device)
        tv_loss = self.tv(self.x)

        for n, m in self.content.items():
            content_loss += m(f[n])

        for n, m in self.style.items():
            style_loss += m(f[n])

        return content_loss, style_loss, tv_loss


def to_tensor(x):
    """
    Arguments:
        x: an instance of PIL image.
    Returns:
        a float tensor with shape [3, h, w],
        it represents a RGB image with
        pixel values in [0, 1] range.
    """
    x = np.array(x)
    x = torch.FloatTensor(x)
    return x.permute(2, 0, 1).unsqueeze(0).div(255.0)
