
# Set the paths to your model checkpoint and tokenizer
ckpt_dir = "/path/to/your/checkpoints"
tokenizer_path = "/path/to/your/tokenizer.json"
max_seq_len = 512  # or whatever sequence length you want
max_batch_size = 8  # or your desired batch size

# Build the Llama instance
llama_model = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

print("Model loaded successfully!")


# ====================================================================
# ====================================================================


# Define the prompts you'd like the model to complete
prompts = ["Once upon a time", "The sun sets over the horizon, and"]

# Generate completions
completions = llama_model.text_completion(
    prompts=prompts,
    temperature=0.7,  # You can adjust this to control randomness
    top_p=0.9,        # Adjust to control diversity
)

# Print the generated completions
for completion in completions:
    print(f"Generated: {completion['generation']}")

# ======================================================================



# Define a dialog for the chatbot
dialogs = [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great, thank you! How about you?"}
    ],
    [
        {"role": "user", "content": "What's the weather like today?"}
    ]
]

# Generate assistant responses
chat_responses = llama_model.chat_completion(
    dialogs=dialogs,
    temperature=0.7,
    top_p=0.9,
)

# Print the assistant's responses
for chat in chat_responses:
    print(f"Assistant: {chat['generation']['content']}")
