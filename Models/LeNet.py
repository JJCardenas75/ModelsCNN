# Importación de los paquetes necesarios
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU
from torch import flatten

class LeNet(Module):
    """
    Implementación de LeNet-5 “al estilo” de tu VGG16:
    - Sin usar torch.nn.functional
    - Con módulos declarados en __init__
    - ReLU y MaxPool instanciados una sola vez
    """
    def __init__(self, nChannels, nclasses):
        super(LeNet, self).__init__()
        
        # Primera convolución (entrada → 6 mapas), kernel 5×5, sin padding
        self.conv1 = Conv2d(in_channels=nChannels,
                            out_channels=6,
                            kernel_size=5,
                            stride=1,
                            padding=0)
        # Segunda convolución (6 → 16 mapas), kernel 5×5, sin padding
        self.conv2 = Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=0)
        
        # Capa de pooling compartida (2×2, stride=2)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        
        # ReLU para activaciones
        self.relu = ReLU(inplace=True)
        
        # Capas totalmente conectadas
        # Tras dos poolings sobre una imagen 32×32 → tamaño espacial = 5×5
        # ⇒ 16·5·5 = 400 entradas
        self.fc1 = Linear(16 * 4 * 4, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, nclasses)

    def forward(self, x):
        # Bloque 1: conv → ReLU → pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Bloque 2: conv → ReLU → pool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Aplanado
        x = flatten(x, 1)   # shape: (batch_size, 16*5*5)
        
        # Clasificador
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x