import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import os

def generate_text(prompt, max_new_tokens=100):
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
    model.load_state_dict(torch.load('./backgammon_model/best_model.pt'))
    model = model.to(device)
    model.eval()
    
    # Encode the prompt
    encoded = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
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

def main():
    print("Backgammon Text Generator")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_input = input("Enter your prompt: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif not user_input:
            continue
        
        # Generate and print response
        try:
            response = generate_text(user_input)
            print(f"\nGenerated text:\n{response}\n")
        except Exception as e:
            print(f"\nError generating text: {e}\n")

if __name__ == "__main__":
    main()
