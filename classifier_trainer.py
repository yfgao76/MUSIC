from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset

# Load dataset
data_files = "" # Path to your dataset file
dataset = load_dataset('csv', data_files=data_files, split='train')
#model_path = "" # If using a pretrained model, set the path here
output_path = "" # Path to save the model

# Create conversations format for the model
def convert_to_conversation_format(examples):
    conversations = []

    for text, lang, label in zip(examples['text'], examples['language'], examples['label']):
        # Create conversation format with user message (text) and model response (label classification)
        conversation = [
            {"role": "user", "content": f":This is a binary classification task. Given the following social media post, predict whether the post contains a mention of an Adverse Drug Event (ADE). Only answer 0 for not mentioning ADE and 1 for mentioning ADE. The post is in {lang}. Post: {text}"},
            {"role": "assistant", "content": f"{label}"}
        ]
        conversations.append(conversation)
    return {"conversations": conversations}

# Apply the conversion
dataset = dataset.map(
    convert_to_conversation_format,
    batched=True,
    remove_columns=['text', 'language', 'label', 'id', 'file_name', 'origin', 'type', 'split']
)

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-12b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 516,
)

# If using pre-trained models, uncomment the following lines
'''
model, tokenizer = FastModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
'''
#preparing dataset
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

dataset = dataset.map(apply_chat_template, batched = True)

#train
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 5, # Train for exactly 1 epoch
        #max_steps = 10,      # Comment out or remove this line
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 2958,
        report_to = "none", # Use this for WandB etc
        output_dir = output_path, # Directory to save model
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# Train the model
trainer.train()

# Save the model explicitly
print(f"Training complete. Saving model to {output_path}...")

# Save the fine-tuned model and tokenizer
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"Model successfully saved to {output_path}")

