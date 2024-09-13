from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# GPT-2 doesn't have a pad token by default, setting it to eos_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.pad_token_id)

# Load the MetaMathQA dataset (which only contains the 'train' split)
dataset = load_dataset("meta-math/MetaMathQA")

# Split the 'train' dataset into training and validation subsets
train_test_split = dataset['train'].train_test_split(test_size=0.2)  # 20% for validation
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']

# Optionally, limit to the first 2000 rows (remove this if you want to use the full dataset)
train_subset = train_dataset.select(range(2000))
validation_subset = validation_dataset.select(range(2000))

# Function to tokenize the dataset
def tokenize_function(examples):
    inputs = examples['query']
    targets = examples['response']
    
    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding='max_length', truncation=True, max_length=128).input_ids

    model_inputs['labels'] = labels
    return model_inputs

# Apply the tokenizer to the subsets
tokenized_train = train_subset.map(tokenize_function, batched=True)
tokenized_validation = validation_subset.map(tokenize_function, batched=True)

# Define the training arguments with the required 'output_dir'
training_args = TrainingArguments(
    output_dir='./results',          # output directory for saving model
    evaluation_strategy='epoch',     # Evaluate at the end of every epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    save_strategy='epoch',
    load_best_model_at_end=True,     # Save and load best model
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation
)

# Train the model
trainer.train()

# Save the model and tokenizer in PyTorch .bin format (this is done by default)
model.save_pretrained('./finetuned_gpt2', safe_serialization=False)  # Saves as pytorch_model.bin
tokenizer.save_pretrained('./finetuned_gpt2')  # Saves tokenizer-related files

# Function to generate a response using GPT-2
def generate_response(prompt):
    # Ensure the model and inputs are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the appropriate device
    model.to(device)
    
    # Tokenize the input and move it to the same device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Generate the response
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage of the generate_response function
if __name__ == "__main__":
    user_input = "What is the Pythagorean theorem?"
    print(generate_response(user_input))
