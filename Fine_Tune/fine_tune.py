import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import PyPDF2
import re
import shutil

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF or text file, cleaning it up and removing images."""
    text = ""
    try:
        # Check if it's a text file
        if pdf_path.endswith('.txt'):
            with open(pdf_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            # Handle PDF files
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

def prepare_dataset(texts, tokenizer):
    # Split text into chunks of max_length
    max_length = 512  # Reduced from 2048 to save memory
    chunks = []
    
    # Split text into smaller pieces before tokenization
    text_pieces = []
    words = texts.split()
    current_piece = []
    current_length = 0
    
    for word in words:
        # More conservative estimate: 2 tokens per word
        word_tokens = len(word.split()) * 2
        if current_length + word_tokens > 1500:  # More conservative limit
            text_pieces.append(' '.join(current_piece))
            current_piece = [word]
            current_length = word_tokens
        else:
            current_piece.append(word)
            current_length += word_tokens
    
    if current_piece:
        text_pieces.append(' '.join(current_piece))
    
    # Process each piece
    for piece in text_pieces:
        # Tokenize the piece
        tokens = tokenizer.encode(piece)
        
        # Create chunks of max_length
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            if len(chunk) == max_length:  # Only use full-length chunks
                chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks of length {max_length}")
    print(f"Processed {len(text_pieces)} text pieces")
    
    # Convert chunks to tensors
    encodings = {
        'input_ids': chunks,
        'attention_mask': [[1] * max_length for _ in chunks],
        'labels': chunks  # Add labels for training
    }
    
    return Dataset.from_dict(encodings)

def fine_tune_tiny_llama():
    # Set memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load the Tiny Llama model and tokenizer
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"  # Automatically handle device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load text files from the downloads folder
    texts = get_all_pdf_texts()
    if texts is None:
        print("Failed to load texts from downloads directory")
        return

    # Prepare the dataset
    dataset = prepare_dataset(texts, tokenizer)
    
    print(f"Created dataset with {len(dataset)} examples")

    # Define training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir='./tmp',  # Minimal temporary directory
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=16,  # Accumulate gradients
        save_strategy="no",  # Disable checkpointing
        run_name="tiny-llama-finetune",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        learning_rate=2e-5,  # Reduced learning rate
        logging_steps=10
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model over the original model
    print("Saving fine-tuned model...")
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    
    # Clean up temporary files
    if os.path.exists('./results'):
        shutil.rmtree('./results')
    
    print("Fine-tuning completed successfully!")
    print(f"Model saved to: {model_name}")

if __name__ == "__main__":
    fine_tune_tiny_llama()
