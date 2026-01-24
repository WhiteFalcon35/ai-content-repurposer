def refined_transcript_prompt(text):
    return f"""
You are an expert educator and thinker.

Extract only what truly matters.

RULES:
- No introductions or conclusions
- No mention of videos, transcripts, or speakers
- Focus on implications, not explanations
- Write 5–7 short paragraphs
- Each paragraph should change how the reader thinks or acts

Content:
{text}
"""


def key_takeaways_prompt(text):
    return f"""
Extract exactly 3–4 key insights.

RULES:
- One sentence per insight
- Each must explain why it matters
- Clear, concise, non-academic

Content:
{text}
"""


def mistakes_prompt(text):
    return f"""
List exactly 3 common mistakes people make related to this topic.

RULES:
- Practical, real-world mistakes
- Clear language
- No academic tone

Content:
{text}
"""


def application_prompt(text):
    return f"""
Convert the core idea into practical behavior.

RULES:
- 2–3 sentences only
- Focus on what to do differently
- Actionable, not motivational

Content:
{text}
"""


def twitter_prompt(text):
    return f"""
You are an experienced professional sharing insight.

Create a Twitter/X thread.

RULES:
- 5–6 tweets
- Each tweet under 25 words
- No emojis, no hashtags
- Clear, confident, experienced tone
- First tweet highlights a common mistake or insight
- Final tweet delivers a strong takeaway

Content:
{text}
"""


def linkedin_prompt(text):
    return f"""
You are writing a thoughtful LinkedIn post.

RULES:
- Calm, professional tone
- Short paragraphs
- One central idea
- End with a practical insight
- No emojis or hashtags

Content:
{text}
"""


def reel_prompt(text):
    return f"""
Generate exactly 3 short hooks for reels or shorts.

RULES:
- Under 10 words each
- Highlight a mistake, cost, or surprising insight
- Direct language
- No emojis, no punctuation

Content:
{text}
"""
