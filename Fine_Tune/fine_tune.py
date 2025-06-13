import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt') or filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def prepare_dataset(texts, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return Dataset.from_dict(encodings)

def fine_tune_tiny_llama():
    # Load the Tiny Llama model and tokenizer
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load text files from the downloads folder
    downloads_dir = './downloads'
    texts = load_text_files(downloads_dir)

    # Prepare the dataset
    dataset = prepare_dataset(texts, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_tiny_llama')
    tokenizer.save_pretrained('./fine_tuned_tiny_llama')

    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    fine_tune_tiny_llama()
