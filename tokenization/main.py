import tiktoken;

encoding = tiktoken.encoding_for_model("gpt-4o")

text = """Hello, how are you doing today? I hope you're having a great day! 
        Let's talk about tokenization and how it works with different models."""

tokens = encoding.encode(text)
print(f"Number of tokens: {len(tokens)}")
print(f"Tokens: {tokens}")

decoded = encoding.decode(tokens)
print(decoded)