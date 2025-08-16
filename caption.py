import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from utils import load_checkpoint
from get_loader import get_loader

def caption_new_image(image_path, model, vocab, device, transform):
    """
    Loads an image, transforms it, and generates a caption.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    # Transform the image and add a batch dimension
    transformed_img = transform(img).unsqueeze(0)
    
    # Generate the caption
    model.eval() # Put model in evaluation mode
    with torch.no_grad():
        caption_indices = model.caption_image(transformed_img.to(device), vocab)
    model.train() # Put model back in training mode
    
    # Convert indices to words and remove special tokens
    caption_words = [word for word in caption_indices if word not in ["<SOS>", "<EOS>", "<PAD>"]]
    
    return " ".join(caption_words)

def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your trained model checkpoint
    checkpoint_path = "final_model.pth.tar" 
    
    # Path to the new image you want to caption
    # IMPORTANT: Change this to your image file name
    new_image_path = "Image_caption_generator/test_examples/child.jpg" 

    # --- Load Model and Vocabulary ---
    # Define transformations (must be same as validation/test)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load vocabulary from the training dataset setup
    # We only need the 'dataset' object to access the vocabulary
    _, dataset = get_loader(
        root_folder="Image_caption_generator/flickr8k/images",
        annotation_file="Image_caption_generator/flickr8k/captions.txt",
        transform=transform,
    )
    vocab = dataset.vocab
    
    # Model hyperparameters (must match your trained model)
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
    
    # Initialize model and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4) # Optimizer is needed for loading checkpoint

    # Load the checkpoint
    try:
        load_checkpoint(torch.load(checkpoint_path, map_location=device,weights_only=True), model, optimizer)
        print(" Model loaded successfully.")
    except FileNotFoundError:
        print(f" Error: Model checkpoint not found at {checkpoint_path}")
        print("Please make sure you have trained the model and saved 'final_model.pth.tar'.")
        return

    # --- Generate and Print Caption ---
    caption = caption_new_image(new_image_path, model, vocab, device, transform)
    
    if caption:
        print("\n--- Image Caption ---")
        print(f"Image: {new_image_path}")
        print(f"Generated Caption: {caption}")
        print("---------------------\n")


if __name__ == "__main__":
    main()