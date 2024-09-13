import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure required NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')

class RuleBasedChatBot:
    def __init__(self):
        # Define some simple rules
        self.rules = {
            "hello": "Hello! How can I assist you today?",
            "hi": "Hi there! What can I do for you?",
            "bye": "Goodbye! Have a great day!",
            "thank": "You're welcome!",
            "thanks": "No problem!"
        }

        # Initialize the NLTK components
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, message):
        # Tokenize the message
        tokens = word_tokenize(message.lower())
        
        # Remove stop words and apply stemming
        processed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return processed_tokens

    def get_response(self, message):
        # Preprocess the message
        processed_message = self.preprocess(message)

        # Check if any of the preprocessed tokens match the predefined rules
        for word in processed_message:
            for key in self.rules:
                if self.stemmer.stem(key) == word:
                    return self.rules[key]

        # Default response if no rules match
        return "I'm not sure how to respond to that. Can you please elaborate?"

# Example usage
if __name__ == "__main__":
    bot = RuleBasedChatBot()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit"]:
            print("Bot: Goodbye!")
            break
        print(f"Bot: {bot.get_response(user_input)}")
