from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        hid = F.relu(self.fc1(x))
        mu = self.fc21(hid)
        logvar = self.fc22(hid)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        stddev = torch.exp(0.5*logvar)
        eps = torch.rand_like(stddev)

        return mu + stddev * eps

    def decode(self, z):
        return F.sigmoid(self.fc4(F.relu(self.fc3(z))))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(input, output, mu, logvar):
    loss = F.binary_cross_entropy(output, input.view(-1, 784), reduction="none")
    import pdb; pdb.set_trace()

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + kl_divergence