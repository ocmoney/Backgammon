from transformers import AutoModelForCausalLM, AutoTokenizer

def download_tiny_llama():
    # Define the model name for Tiny Llama
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

    # Download the model and tokenizer
    print(f"Downloading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer locally
    model.save_pretrained('./tiny_llama_model')
    tokenizer.save_pretrained('./tiny_llama_model')

    print("Download and save completed successfully!")

if __name__ == "__main__":
    download_tiny_llama()
