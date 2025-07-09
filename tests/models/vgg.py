import pdb
import torch
import torch.nn as nn

# fmt: off
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg-like': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg-like-tinyimagenet': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# fmt: on


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()

        self.cfg = cfg[vgg_name]
        self.num_classes = num_classes
        self.vgg_name = vgg_name

        self.features, self.features_masks = self._make_layers(cfg[vgg_name])

        if "like" not in vgg_name:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, self.num_classes),
            )

            self.classifier_masks = [
                torch.ones(512, 512),
                torch.ones(512),
                torch.ones(512, 512),
                torch.ones(512),
                torch.ones(self.num_classes, 512),
                torch.ones(self.num_classes),
            ]
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, self.num_classes),
            )

            self.classifier_masks = [
                torch.ones(512, 512),
                torch.ones(512),
                torch.ones(self.num_classes, 512),
                torch.ones(self.num_classes),
            ]

        self.apply(_weights_init)

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), out.size(1) * out.size(2) * out.size(3))
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        masks = []
        in_channels = 3
        flag_tinyimagenet = "tinyimagenet" in self.vgg_name
        for x in cfg:
            if flag_tinyimagenet:
                # Stride 2 for tinyimagenet for first layer
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
                flag_tinyimagenet = False
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                masks += [torch.ones(x, in_channels, 3, 3), torch.ones(x)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
        return nn.Sequential(*layers), masks


def get_vgglike(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return VGG(vgg_name="vgg-like", num_classes=10)
    elif dataset == "tinyimagenet":
        return VGG(vgg_name="vgg-like-tinyimagenet", num_classes=200)
    raise NotImplementedError(f"LeNet is not available for {dataset}")


def get_vgglike_trained(dataset: str) -> nn.Module:
    if dataset == "tinyimagenet":
        path = "/home/think-server/Documents/cpn/checkpoints_all_results/vgglike_tinyimagenet_noprune/model.trained.ckpt"
        model = VGG(vgg_name="vgg-like-tinyimagenet", num_classes=200)
        model.load_state_dict(torch.load(path)["model"])
        return model
    elif dataset == "cifar10":
        path = "/home/think-server/Documents/cpn/checkpoints_all_results/vgglike_cifar10_noprune/model.trained.ckpt"
        model = VGG(vgg_name="vgg-like", num_classes=10)
        model.load_state_dict(torch.load(path)["model"])
        return model
    else:
        raise NotImplementedError(f"VGG-like is not available for {dataset}")
