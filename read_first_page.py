import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pdfplumber

def print_pdf_first_page(pdf_path, title):
    """Print the first page of a PDF file"""
    try:
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            print("Make sure to run the download script first to get the PDF.")
            return
        
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                print("PDF file appears to be empty")
                return
            
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            print("=" * 80)
            print(f"FIRST PAGE OF {title}")
            print("=" * 80)
            print(text)
            print("=" * 80)
            print(f"Total pages in PDF: {len(pdf.pages)}")
            print("=" * 80)
        
    except Exception as e:
        print(f"Error reading PDF: {e}")

def load_model_and_tokenizer():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = GPT2Config(
        vocab_size=50257,
        n_positions=256,
        n_ctx=256,
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
        use_cache=True
    )
    
    # Load tokenizer and create model with correct config
    tokenizer = GPT2Tokenizer.from_pretrained('./backgammon_model')
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load('./backgammon_model/model.pt'))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer, device

def get_word_embedding(word, model, tokenizer, device):
    # Tokenize the word
    inputs = tokenizer(word, return_tensors='pt').to(device)
    
    # Get the embedding from the model's word embeddings
    with torch.no_grad():
        # Get the input embedding
        word_embedding = model.transformer.wte(inputs['input_ids'])
        # Average the embedding across the sequence length
        word_embedding = word_embedding.mean(dim=1)
    
    return word_embedding.cpu().numpy()

def find_similar_words(target_word, model, tokenizer, device, top_k=5):
    # Get the target word's embedding
    target_embedding = get_word_embedding(target_word, model, tokenizer, device)
    
    # Get embeddings for all words in the vocabulary
    vocab_size = tokenizer.vocab_size
    all_embeddings = []
    word_ids = []
    
    print(f"Computing embeddings for vocabulary...")
    for word_id in range(vocab_size):
        word = tokenizer.decode([word_id])
        if len(word.strip()) > 0:  # Skip empty tokens
            embedding = get_word_embedding(word, model, tokenizer, device)
            all_embeddings.append(embedding)
            word_ids.append(word_id)
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(target_embedding, all_embeddings)[0]
    
    # Get top k similar words
    top_indices = np.argsort(similarities)[-top_k-1:-1][::-1]  # -1 to exclude the word itself
    
    similar_words = []
    for idx in top_indices:
        word = tokenizer.decode([word_ids[idx]])
        similarity = similarities[idx]
        similar_words.append((word, similarity))
    
    return similar_words

def main():
    # Print first pages of both PDFs
    pdf_path1 = os.path.join('downloads', 'GNUBackgammon_vs_Mamoun.pdf')
    pdf_path2 = os.path.join('downloads', 'BackGammon.pdf')
    
    print_pdf_first_page(pdf_path1, "GNUBackgammon_vs_Mamoun.pdf")
    print("\n")
    print_pdf_first_page(pdf_path2, "BackGammon.pdf")
    
    print("\n" + "="*80)
    print("STARTING MODEL ANALYSIS")
    print("="*80 + "\n")
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Words to find similarities for
    target_words = ['double', 'move', 'opponent', 'useful', '13/7', 'point']
    
    # Find and print similar words for each target word
    # for word in target_words:
    #     print(f"\nTop 10 words similar to '{word}':")
    #     print("-" * 50)
    #     similar_words = find_similar_words(word, model, tokenizer, device)
    #     for similar_word, similarity in similar_words:
    #         print(f"{similar_word}: {similarity:.4f}")

if __name__ == "__main__":
    main()