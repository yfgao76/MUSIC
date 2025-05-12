from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
import pandas as pd

# Define paths
model_path = ""
test_data_path = ""
output_path = ""

print(f"Loading fine-tuned model from {model_path}...")

# Load the fine-tuned model
model, tokenizer = FastModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    # Use the same settings as the training script
)

# Set up the tokenizer with the right chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# Load test dataset
test_dataset = load_dataset('csv', data_files=test_data_path, split='train')
print(f"Loaded {len(test_dataset)} examples for inference")

# Create list to store results
results = []

# Perform inference on each example
for i, example in enumerate(test_dataset):
    # Create the prompt in the same format as during training
    messages = [{
        "role": "user", 
        "content": f":This is a binary classification task. Given the following social media post, predict whether the post contains a mention of an Adverse Drug Event (ADE). Only answer 0 for not mentioning ADE and 1 for mentioning ADE. The post is in {example['language']}. Post: {example['text']}"
    }]
    
    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )
    
    # Generate prediction with scores
    with torch.no_grad():
        # Modified to return scores and detailed output
        outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_new_tokens=8,  # Short for classification outputs
            temperature=0.1,   # Lower temperature for more deterministic results
            top_p=0.95, 
            top_k=64,
            return_dict_in_generate=True,  # Return detailed output dict
            output_scores=True,            # Return scores for each token
        )
    
    # Get the generated token IDs and scores
    generated_ids = outputs.sequences[0]
    # Get the input length to ignore prompt tokens
    input_length = tokenizer([text], return_tensors="pt").input_ids.shape[1]
    # Get only the generated part (excluding input)
    generated_part = generated_ids[input_length:]
    
    # Decode the output for binary prediction
    decoded_output = tokenizer.decode(generated_part)
    
    # Find the first occurrence of 0 or 1
    prediction = None
    for char in decoded_output:
        if char in ["0", "1"]:
            prediction = char
            break
    
    if prediction is None:
        print(f"Warning: Could not extract prediction for example {i}. Using default '0'")
        prediction = "0"

    # Store result
    results.append({
        "id": example.get("id", str(i)),
        "language": example["language"],
        "text": example["text"],
        "prediction": prediction,
        "label": str(example.get("label", "")) if "label" in example else "",
    })
    
    # Print progress
    if (i + 1) % 10 == 0 or i == len(test_dataset) - 1:
        print(f"Processed {i+1}/{len(test_dataset)} examples")

# Save predictions to CSV
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
