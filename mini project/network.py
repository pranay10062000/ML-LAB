from model import *

# Neural Network
class Network(nn.Module):
    def __init__(self,size_val):
        super().__init__()
        self.Encoder = nn.Sequential(
                nn.Linear(size_val,size_val//2),
                nn.Tanh(),
                nn.Linear(size_val//2,size_val//4),
                nn.Tanh(),
                nn.Linear(size_val//4,size_val//8),
                nn.Tanh(),
                nn.Linear(size_val//8,32),
                nn.Tanh()
        )
        self.Decoder = nn.Sequential(
                nn.Linear(32,size_val//8),
                nn.Tanh(),
                nn.Linear(size_val//8,size_val//4),
                nn.Tanh(),
                nn.Linear(size_val//4,size_val//2),
                nn.Tanh(),
                nn.Linear(size_val//2,size_val),
                nn.Tanh()
        )
    def forward(self,x):
        x=self.Encoder(x)
        x=self.Decoder(x)
        return x
    
EdgeDetector = Network(img_size*img_size).to(device)
EdgeDetector.load_state_dict(torch.load("D:\\ML Mini project\\weights1.pth"))
