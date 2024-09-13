import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_gpt2')
model = GPT2LMHeadModel.from_pretrained('./finetuned_gpt2')

# Function to generate a response using GPT-2
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to handle sending messages
def send_message():
    user_message = user_input.get()
    if user_message.strip() != "":
        # Display user message on the right
        display_message(user_message, is_user=True)
        # Generate GPT-2 bot response and display it on the left
        bot_response = generate_response(user_message)
        display_message(bot_response, is_user=False)
        # Clear the input field
        user_input.delete(0, tk.END)

def display_message(message, is_user):
    chat_window.configure(state='normal')
    if is_user:
        chat_window.insert(tk.END, f"\nYou: {message}\n")
    else:
        chat_window.insert(tk.END, f"\nBot: {message}\n")
    chat_window.configure(state='disabled')
    chat_window.yview(tk.END)  # Auto-scroll to the bottom

# Initialize the main window
root = tk.Tk()
root.title("GPT-2 Chatbot")
root.geometry("400x500")

# Create a chat window (scrollable text widget)
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Helvetica", 12))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a frame for input and button
input_frame = tk.Frame(root)
input_frame.pack(fill=tk.X, padx=10, pady=10)

# Create an entry widget for user input
user_input = tk.Entry(input_frame, width=80, font=("Helvetica", 12))
user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

# Create a send button
send_button = tk.Button(input_frame, text="Send", command=send_message, width=10)
send_button.pack(side=tk.RIGHT)

# Function to bind Enter key to send messages
def on_enter_key(event):
    send_message()

root.bind('<Return>', on_enter_key)

# Run the main loop
root.mainloop()
