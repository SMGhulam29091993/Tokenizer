from huggingface_hub import get_token
from openai import OpenAI
from transformers import pipeline


MODELS = "google/gemma-4-31B-it:novita"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=get_token(),
)


pipe = pipeline("image-text-to-text", )
completion = client.chat.completions.create(
    model= MODELS,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one sentence. What is that big statue?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                    }
                }
            ]
        }
    ],
)

print(pipe(completion.choices[0].message))