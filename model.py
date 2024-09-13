from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import optuna

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the dataset
dataset = load_dataset("meta-math/MetaMathQA")

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",     # Evaluate every x steps
    eval_steps=500,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],  # Optional: Add validation for better tuning
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./finetuned_gpt2')
tokenizer.save_pretrained('./finetuned_gpt2')

def train_model(learning_rate, batch_size, num_train_epochs):
    # Update training arguments with Optuna suggested parameters
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_dir='./logs_optuna',
        logging_steps=10,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    # Re-initialize Trainer with the updated arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )
    
    # Train the model
    trainer.train()
    
    # Return evaluation loss
    eval_results = trainer.evaluate()
    return eval_results['eval_loss']

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)

    # Train the model with the suggested hyperparameters
    eval_loss = train_model(learning_rate, batch_size, num_train_epochs)
    return eval_loss

# Create a study object and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Number of trials to run

# Print the best hyperparameters found
print("Best hyperparameters:", study.best_params)
print("Best trial evaluation loss:", study.best_value)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage of the generate_response function
if __name__ == "__main__":
    user_input = "Hello, how can I help you today?"
    print(generate_response(user_input))
