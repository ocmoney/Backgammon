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

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, cleaning it up and removing images."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                # Clean up the text
                page_text = re.sub(r'\s+', ' ', page_text)  # Remove extra whitespace
                page_text = re.sub(r'[^\w\s.,!?-]', '', page_text)  # Keep only basic punctuation
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

class BackgammonDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256):
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
        
        print(f"Created {len(self.chunks)} text chunks for training")
        
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
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    pdf_path = 'downloads/GNUBackgammon_vs_Mamoun.pdf'
    text = extract_text_from_pdf(pdf_path)
    
    if text is None:
        print("Failed to extract text from PDF")
        return
    
    print(f"Extracted {len(text.split())} words from PDF")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = BackgammonDataset(text, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 10
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
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
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_dir = './backgammon_model'
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Save model state dict
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint at epoch {epoch + 1}")
    
    # Save final model
    print("Saving final model and tokenizer...")
    checkpoint_dir = './backgammon_model'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
    tokenizer.save_pretrained(checkpoint_dir)
    
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
    train_model()
    
    # Example of generating text
    prompt = "In backgammon, the key strategy is"
    generated_text = generate_text(prompt)
    print("\nGenerated text:")
    print(generated_text)
