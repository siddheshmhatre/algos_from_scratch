from cProfile import label
import torch
import enum

def train_one_epoch(model, loss_function, optimizer, dataloader, log_frequency, epoch, device):
    model.train()
    train_loss = 0.0
    num_iters = 0
    
    for idx, (data, labels) in enumerate(dataloader):
        num_iters += 1
        optimizer.zero_grad()
        data = data.to(device)
        labels = labels.to(device)
        output = model(data.view(data.shape[0], 784))
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if idx % log_frequency == 0:
            print (f"Epoch: {epoch} Current batch loss: {loss.item()} Aggr Train loss: {train_loss / num_iters}")

    return train_loss / num_iters

def test(model, loss_function, dataloader, log_frequency, epoch, device):
    model.eval()
    test_loss = 0.0
    num_iters = 0
    total_incorrect = 0
    total = 0

    for idx, (data, labels) in enumerate(dataloader):
        total += data.size(0)
        num_iters += 1
        data = data.to(device)
        labels = labels.to(device)

        output = model(data.view(data.shape[0], 784))
        loss = loss_function(output, labels)

        test_loss += loss.item()

        incorrect = (output.argmax(dim=1) != labels).sum().item()
        total_incorrect += incorrect

        if idx % log_frequency == 0:
            print (f"Epoch: {epoch} Current batch loss: {loss.item()} Aggr Train loss: {test_loss / num_iters}")

    print (f"Total misses : {total_incorrect} / {total} Average error : {total_incorrect / total}")

    return test_loss / num_iters, total_incorrect, total