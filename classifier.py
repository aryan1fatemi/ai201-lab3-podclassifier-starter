import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)

def load_labeled_examples() -> list[dict]:
    # [Existing code remains unchanged]
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled

def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    prompt = (
        "You are a podcast classification AI. Your task is to classify podcast episode "
        "descriptions into exactly one of four categories based on their format:\n"
        "- 'interview': A host speaks with one or more guests. Structured around Q&A.\n"
        "- 'solo': One host speaks alone, without guests.\n"
        "- 'panel': Three or more speakers discuss a topic together as rough equals.\n"
        "- 'narrative': A story is told over the episode, assembled from reporting, clips, or multiple sources.\n\n"
        "Here are some examples of correctly labeled episodes to learn from:\n\n"
    )

    # Inject the few-shot examples
    for ex in labeled_examples:
        prompt += f"Title: {ex['title']}\nDescription: {ex['description']}\nLabel: {ex['label']}\n\n"

    # Present the new task and strict output instructions
    prompt += (
        "Now, classify the following new episode description.\n\n"
        f"Description: {description}\n\n"
        "Output your response strictly as a raw JSON object with exactly two keys: 'label' and 'reasoning'.\n"
        "The 'label' value MUST be exactly one of: 'interview', 'solo', 'panel', or 'narrative'.\n"
        "Do not include markdown formatting, backticks, or any other text outside the JSON object."
    )
    
    return prompt

def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise text-classification assistant that only outputs valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0 # Force deterministic output for classification tasks
        )
        
        text = response.choices[0].message.content.strip()

        # Strip markdown formatting just in case the LLM ignores instructions
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse the JSON response
        parsed_response = json.loads(text)
        label = parsed_response.get("label", "").strip()
        reasoning = parsed_response.get("reasoning", "No reasoning provided.")

        # Validate label
        if label not in VALID_LABELS:
            label = "unknown"

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        # Gracefully handle API failures or JSON decoding errors
        return {
            "label": "unknown",
            "reasoning": f"Classification failed due to an error: {str(e)}"
        }