# Modified train.py

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# NEW: Import matplotlib and the new utility function
import matplotlib.pyplot as plt
from utils import save_checkpoint, load_checkpoint, get_examples_as_text 
from get_loader import get_loader
from model import CNNtoRNN


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="Image_caption_generator/flickr8k/images",
        annotation_file="Image_caption_generator/flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 20

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()
    
    # NEW: Lists to store loss and epoch-wise outputs
    all_epoch_losses = []
    all_epoch_outputs = []

    for epoch in range(num_epochs):
        # Get and store example outputs at the start of the epoch
        example_output = get_examples_as_text(model, device, dataset)
        all_epoch_outputs.append(f"--- EPOCH {epoch + 1} ---\n{example_output}\n")
        
        # NEW: Variable to track loss for the current epoch
        running_loss = 0.0

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            
            # NEW: Add batch loss to the running epoch loss
            running_loss += loss.item()

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # NEW: Calculate and store the average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        all_epoch_losses.append(epoch_loss)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


    # NEW: Save the final model checkpoint AFTER all epochs are done
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="final_model.pth.tar")

    # NEW: Save the collected outputs to a text file
    with open("training_outputs.txt", "w") as f:
        for output in all_epoch_outputs:
            f.write(output + "\n")
    print("Saved all epoch outputs to training_outputs.txt")

    # NEW: Save the epoch losses to a text file
    with open("epoch_losses.txt", "w") as f:
        for i, loss in enumerate(all_epoch_losses):
            f.write(f"Epoch {i+1}: {loss}\n")
    print("Saved all epoch losses to epoch_losses.txt")

    # NEW: Plot the loss vs. epoch graph and save it
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), all_epoch_losses, marker='o')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig('graph.png')
    print("Saved loss graph to graph.png")


if __name__ == "__main__":
    train()