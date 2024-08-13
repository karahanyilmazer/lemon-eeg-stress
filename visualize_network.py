# %%
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot


class ModifiedEEGNet(nn.Module):
    def __init__(
        self,
        chans=19,  # Updated channel count
        time_points=640,  # Updated time points
        f1=16,  # Increased initial filter count
        f2=32,  # Increased output filter count for block3
        d=4,  # Increased depth multiplier
        dropoutRate=0.5,
        max_norm1=1,
        max_norm2=0.25,
    ):
        super(ModifiedEEGNet, self).__init__()

        # Adjusted FC input feature calculation based on pool and conv layers
        linear_input_size = f2 * 5  # Correct input size for fc1 after Block 4

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(
                1, f1, (1, 64), padding='same', bias=False
            ),  # Increased kernel size
            nn.BatchNorm2d(f1),
        )

        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),  # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                d * f1,
                f2,
                (1, 32),
                groups=f2,
                bias=False,
                padding='same',  # Increased kernel size
            ),  # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),  # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate),
        )

        # Optional additional convolutional block for increased complexity
        self.block4 = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )

        self.flatten = nn.Flatten()

        # Change the output of the fully connected layer to have more neurons
        self.fc1 = nn.Linear(linear_input_size, 256)  # Correct input feature size
        self.fc2 = nn.Linear(256, 1)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layers
        self._apply_max_norm(self.fc1, max_norm2)
        self._apply_max_norm(self.fc2, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # Optional: include this line if block4 is added
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


device = 'cpu'
model = ModifiedEEGNet().to(device)

# Load the model weights
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path, weights_only=False))

# Set the model to evaluation mode
model.eval()

dummy_input = Variable(torch.randn(1, 1, 19, 640)).to(device)

# Generate the graph
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.graph_attr.update({'dpi': '300'})

for node in dot.body:
    if 'Conv2d' in node:
        node = node.replace('color=black', 'style=filled color=lightgreen')
    elif 'BatchNorm2d' in node:
        node = node.replace('color=black', 'style=filled color=lightyellow')
    elif 'Linear' in node:
        node = node.replace('color=black', 'style=filled color=lightpink')
    elif 'ELU' in node:
        node = node.replace('color=black', 'style=filled color=lightblue')
    elif 'AvgPool2d' in node:
        node = node.replace('color=black', 'style=filled color=lightcyan')
    elif 'Dropout' in node:
        node = node.replace('color=black', 'style=filled color=lightgray')
    else:
        node = node.replace('color=black', 'style=filled color=white')

dot.format = 'png'
dot.render('model_graph')

# %%
