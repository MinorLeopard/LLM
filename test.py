from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Define the path where the model is saved
model_path = 'finetuned_llama3'  # Update this to the path where you saved the model

# Check for GPU/CPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Limit the number of threads used by PyTorch
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded.")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True).to(device)
    print("Model loaded.")

    # Example usage
    input_text = "What is the latest price of TATA Motors?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate a response from the model
    print("Generating response...")
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    print("Response generated.")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Decoded generated text.")

    print(f"Generated text: {generated_text}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
