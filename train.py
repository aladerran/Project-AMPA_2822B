# train.py

import sys
import time
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
from models import ResNet9
from simple_ml import train_cifar10, evaluate_cifar10

def main():
    device = ndl.cuda()
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
    )
    model = ResNet9(device=device, dtype="float32")

    start_time = time.time()
    train_cifar10(
        model, 
        dataloader, 
        n_epochs=10, 
        optimizer=ndl.optim.Adam,
        lr=0.001, 
        weight_decay=0.001
    )
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    start_time = time.time()
    evaluate_cifar10(model, dataloader)
    eval_time = time.time() - start_time
    print(f"Evaluation time: {eval_time:.2f} seconds")

if __name__ == "__main__":
    main()
