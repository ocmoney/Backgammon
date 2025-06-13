from transformers import AutoModelForCausalLM, AutoTokenizer

def run_tiny_llama(prompt, max_length=100):
    # Load the Tiny Llama model and tokenizer
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate text
    outputs = model.generate(
        inputs['input_ids'],
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

def main():
    print("Tiny Llama Text Generator")
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
            response = run_tiny_llama(user_input)
            print(f"\nGenerated text:\n{response}\n")
        except Exception as e:
            print(f"\nError generating text: {e}\n")

if __name__ == "__main__":
    main()
