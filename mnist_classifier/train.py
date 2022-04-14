import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Model
from utils import train_one_epoch, test
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Some constants as params for training
BATCH_SIZE = 128
NUM_WORKERS = 12
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
LOG_FREQ = 50

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the datasets
    train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transforms.ToTensor())

    # Create data loader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # Create model
    model = Model().to(device)

    # Loss function
    loss_func = F.cross_entropy

    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    min_test_loss = float('inf')
    min_number_of_misses = float('inf')
    min_loss_epoch = None
    min_num_misses_epoch = None

    # Run training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, loss_func, optimizer, train_dataloader, LOG_FREQ, epoch, device)

        print (f"Epoch: {epoch} Average Train loss: {train_loss}")

        test_loss, incorrect, num_samples = test(model, loss_func, test_dataloader, LOG_FREQ, epoch, device)

        if test_loss < min_test_loss:
            # Save the model state dict
            # For now print
            min_test_loss = test_loss
            min_loss_epoch = epoch
            print(f"{test_loss} is new minimum test loss at epoch {epoch}")

        if incorrect < min_number_of_misses:
            min_number_of_misses = incorrect
            min_num_misses_epoch = epoch
            print(f"{min_number_of_misses}/{num_samples} is new minimum number of misses at epoch {epoch}")

    print (f"Min Test loss {min_test_loss} at epoch {min_loss_epoch}")
    print (f"Min number of misses {min_number_of_misses}/{num_samples} at epoch {min_num_misses_epoch}")

if __name__ == "__main__":
    main()

