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


def explain_image_with_context(image_description, transcript):
    prompt = f"""
You are helping a beginner understand a video.

The speaker is saying:
{transcript}

Explain what the image likely represents.
Explain it simply.
Why does it matter in the video?
"""
    return generate_text(prompt)
