import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained BARD model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

# Define a function to generate responses
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Define a function to handle user input
def handle_input():
    user_input = input("You: ")
    response = generate_response(user_input)
    print("Bot: " + response)

# Start the chatbot
print("Hey there! I'm Twinkle, your chat bot companion. Is there anything that I can help you with today?")
print("Type 'exit' to end the conversation.")
while True:
    handle_input()
    if user_input.lower() == "exit":
        break
