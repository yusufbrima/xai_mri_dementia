import torch
import torch.nn as nn
import torch.nn.functional as F


class Small3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Small3DCNN, self).__init__()
        
        # First Block: Convolutional Layers with reduced filters
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)  # in_channels, out_channels
        self.pool1 = nn.MaxPool3d(2)
        self.bn1 = nn.BatchNorm3d(16)  # Batch Normalization
        
        # Second Block: Convolutional Layers with reduced filters
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.bn2 = nn.BatchNorm3d(32)  # Batch Normalization
        
        # Third Block: Convolutional Layers with reduced filters
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.bn3 = nn.BatchNorm3d(64)  # Batch Normalization
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(1032192, 128)  # Adjusted input size for fc1
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim == 4:  # [B, D, H, W]
            x = x.unsqueeze(1)  # Add channel dimension: [B, 1, D, H, W]

        # Forward pass through the convolutional blocks with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # Now with (B, 64, D', H', W')

        # Flatten to feed into fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to [B, 64 * D' * H' * W']

        # Fully connected layers
        x = F.relu(self.fc1(x))
        out = self.fc2(x)  # Output layer: [B, num_classes]

        return out

# --- DenseNet3D Implementation (Further Reduced) ---
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=2, drop_rate=0.0):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout3d(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=2, drop_rate=0.0):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)

class DenseNet3D(nn.Module):
    def __init__(
        self,
        num_init_features=4,       # smaller init features
        growth_rate=4,             # smaller growth rate
        block_config=(1, 1, 1),    # only 1 layer per block
        bn_size=2,
        compression=0.5,
        drop_rate=0.0,
        num_classes=3
    ):
        super(DenseNet3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i < len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        features = self.features(x)
        out = F.relu(features)
        out = F.adaptive_avg_pool3d(out, (1,1,1)).view(x.size(0), -1)
        return self.classifier(out)



# --- ResNet3D Implementation ---
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(ResNet3D, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv3d(1, 16, kernel_size=7, stride=(1,2,2), padding=(3,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # just pass 
    pass