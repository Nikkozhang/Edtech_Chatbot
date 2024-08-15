# chatbot.py

class RuleBasedChatBot:
    def __init__(self):
        # Define some simple rules
        self.rules = {
            "hello": "Hello! How can I assist you today?",
            "hi": "Hi there! What can I do for you?",
            "bye": "Goodbye! Have a great day!",
            "thank you": "You're welcome!",
            "thanks": "No problem!"
        }

    def get_response(self, message):
        # Convert the message to lowercase to make matching case-insensitive
        message = message.lower()

        # Check if the message matches any of the predefined rules
        for key in self.rules:
            if key in message:
                return self.rules[key]

        # Default response if no rules match
        return "I'm not sure how to respond to that. Can you please elaborate?"
