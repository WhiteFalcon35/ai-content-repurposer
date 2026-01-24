from openai import OpenAI

client = OpenAI()

def generate_text(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return response.output_text.strip()
