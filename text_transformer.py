import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config
)
import os
from tqdm import tqdm
import PyPDF2
import re
import wandb
import numpy as np

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF or text file, cleaning it up and removing images."""
    text = ""
    try:
        # Check if it's a text file
        if pdf_path.endswith('.txt'):
            with open(pdf_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            # Handle PDF files as before
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    # Clean up the text
                    page_text = re.sub(r'\s+', ' ', page_text)  # Remove extra whitespace
                    page_text = re.sub(r'[^\w\s.,!?-]', '', page_text)  # Keep only basic punctuation
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return text

def get_all_pdf_texts():
    """Get text from all PDFs and text files in the downloads directory"""
    all_text = ""
    downloads_dir = 'downloads'
    
    if not os.path.exists(downloads_dir):
        print(f"Downloads directory not found: {downloads_dir}")
        return None
    
    # Get all PDF and text files in the downloads directory
    files = [f for f in os.listdir(downloads_dir) if f.endswith(('.pdf', '.txt'))]
    
    if not files:
        print("No PDF or text files found in downloads directory")
        return None
    
    print(f"Found {len(files)} files")
    
    # Process each file
    for file in files:
        file_path = os.path.join(downloads_dir, file)
        print(f"\nProcessing {file}...")
        
        text = extract_text_from_pdf(file_path)
        if text:
            print(f"Successfully extracted {len(text.split())} words from {file}")
            all_text += text + "\n\n"
        else:
            print(f"Failed to extract text from {file}")
    
    return all_text

class BackgammonDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256, is_training=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Split text into chunks of max_length
        self.chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Tokenize the word to get its length
            word_tokens = len(self.tokenizer.encode(word))
            if current_length + word_tokens > max_length:
                # Save current chunk and start new one
                self.chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens
        
        if current_chunk:
            self.chunks.append(' '.join(current_chunk))
        
        # Calculate split points for train/test
        total_chunks = len(self.chunks)
        test_size = int(total_chunks * 0.1)  # 10% for testing
        middle_start = total_chunks // 2 - test_size // 2
        
        if is_training:
            # For training, use chunks before and after the test set
            self.chunks = self.chunks[:middle_start] + self.chunks[middle_start + test_size:]
        else:
            # For testing, use chunks from the middle
            self.chunks = self.chunks[middle_start:middle_start + test_size]
        
        print(f"Created {len(self.chunks)} text chunks for {'training' if is_training else 'testing'}")
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        encodings = self.tokenizer(
            chunk,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'][0],
            'attention_mask': encodings['attention_mask'][0]
        }

def train_model():
    # Initialize wandb
    wandb.init(
        project="backgammon-gpt",
        config={
            "architecture": "GPT2",
            "dataset": "Backgammon PDFs",
            "epochs": 20,
            "batch_size": 4,
            "learning_rate": 3e-4,
            "max_length": 256,
            "n_layer": 4,
            "n_head": 12,
            "n_embd": 768
        }
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configuration
    print("Creating model configuration...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=256,        # Reduced from 512
        n_ctx=256,             # Reduced from 512
        n_embd=768,            # Standard GPT-2 embedding dimension
        n_layer=4,             # Reduced from 12
        n_head=12,             # Standard GPT-2 number of heads
        n_inner=3072,          # Standard GPT-2 inner dimension
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    model = GPT2LMHeadModel(config).to(device)
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {model_size:.2f}M parameters")
    wandb.log({"model_size_millions": model_size})
    
    # Extract text from all PDFs
    print("Extracting text from PDFs...")
    text = get_all_pdf_texts()
    
    if text is None:
        print("Failed to extract text from PDFs")
        return
    
    total_words = len(text.split())
    print(f"Extracted {total_words} total words from all PDFs")
    wandb.log({"total_words": total_words})
    
    # Create training and testing datasets
    print("Creating datasets...")
    train_dataset = BackgammonDataset(text, tokenizer, is_training=True)
    test_dataset = BackgammonDataset(text, tokenizer, is_training=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 10
    
    # Early stopping setup
    best_test_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log batch loss
            wandb.log({"batch_loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}')
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                test_loss += outputs.loss.item()
        
        avg_test_loss = test_loss / len(test_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Test Loss: {avg_test_loss:.4f}')
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "test_loss": avg_test_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Early stopping check
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            checkpoint_dir = './backgammon_model'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved best model at epoch {epoch + 1} with test loss: {best_test_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        model.train()
        
        # Save regular checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_dir = './backgammon_model'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt'))
            print(f"Saved checkpoint at epoch {epoch + 1}")
    
    # Save final model
    print("Saving final model and tokenizer...")
    checkpoint_dir = './backgammon_model'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pt'))
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Close wandb
    wandb.finish()
    
    print("Training completed!")

def generate_text(prompt, max_length=100):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./backgammon_model')
    
    # Create model with standard configuration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=256,
        n_ctx=256,
        n_embd=768,
        n_layer=4,
        n_head=12,
        n_inner=3072,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load('./backgammon_model/model.pt'))
    model = model.to(device)
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Train the model
    print("Starting model training...")
    train_model()
    
    # Only try to generate text if the model exists
    model_path = './backgammon_model/model.pt'
    if os.path.exists(model_path):
        print("\nGenerating sample text...")
        prompt = "In backgammon, the key strategy is"
        generated_text = generate_text(prompt)
        print("\nGenerated text:")
        print(generated_text)
    else:
        print("\nModel training completed. To generate text, run the script again after training is complete.")
