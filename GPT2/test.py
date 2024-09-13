# import json

# # Path to the tokenizer config file
# file_path = './finetuned_gpt2/tokenizer_config.json'

# try:
#     with open(file_path, 'r') as f:
#         json_data = json.load(f)
#     print("JSON is valid!")
# except json.JSONDecodeError as e:
#     print(f"Error parsing JSON: {e}")
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # Load the pretrained GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# # Generate a sample response
# inputs = tokenizer("Hello, how are you?", return_tensors='pt')
# outputs = model.generate(inputs['input_ids'], max_length=50)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(response)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer and model from the custom directory
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_gpt2')
model = GPT2LMHeadModel.from_pretrained('./finetuned_gpt2')

# Test the tokenizer and model by generating a response
inputs = tokenizer("What is the capital of France?", return_tensors='pt')
outputs = model.generate(inputs['input_ids'], max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
