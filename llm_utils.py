from openai import OpenAI

client = OpenAI()

def generate_text(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    # Safely extract text output
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    # Fallback (very important for reliability)
    for item in response.output:
        if item["type"] == "output_text":
            return item["text"].strip()

    return ""
