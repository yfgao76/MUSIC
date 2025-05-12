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

#get the first 100 rows for testing
#test_dataset = test_dataset.select(range(100))

# Create list to store results
results = []

# Get token IDs for "0" and "1" for later probability extraction
# Replace encode method with direct tokenizer call

# Perform inference on each example
for i, example in enumerate(test_dataset):
    text = example['text']
    lang = example['language']
    pred = example['prediction']
    # Create conversation format with user message (text) and model response (label classification)     
    if pred == 0:
        messages = [
            {"role": "user", "content": f": The following social media post was classified as not containing a mention of an Adverse Drug Event (ADE). Criticize on whether the classification is right. Give your answer 0 if you think the classification is wrong and 1 if you think it is right at the end. The post is in {lang}. Post: {text}"},
        ]
    elif pred == 1:
        messages = [
            {"role": "user", "content": f": The following social media post was classified as containing a mention of an Adverse Drug Event (ADE). Criticize on whether the classification is right. Give your answer 0 if you think the classification is wrong and 1 if you think it is right at the end. The post is in {lang}. Post: {text}"},
        ]      

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
            max_new_tokens=2048,  # Short for classification outputs
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
    
    # Find the last occurrence of 0 or 1
    prediction = None
    for char in reversed(decoded_output):
        if char in ["0", "1"]:
            prediction = char
            break
    if prediction is None:
        print(f"Warning: Could not extract prediction for example {i}. Using default '1'")
        prediction = "1"
    
    # Store result
    results.append({
        "id": example.get("id", str(i)),
        #"language": example["language"],
        #"text": example["text"],
        "predicted_label": 1 - int(prediction) ^ pred,
        #"label": str(example.get("label", "")) if "label" in example else "",
        #"reasoning": decoded_output
    })
    
    # Print progress
    if (i + 1) % 10 == 0 or i == len(test_dataset) - 1:
        print(f"Processed {i+1}/{len(test_dataset)} examples")
    
    #if (i > 100):
        #break

# Save predictions to CSV
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

