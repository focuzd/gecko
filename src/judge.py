import json
import os
import re
import time
import urllib.error
import urllib.request

# both speak the OpenAI chat completions API, so switching is just a different url + key
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

CRITERIA = {
    "spelling": "1 = mostly non-words, 3 = many misspelled or made-up words, 5 = no spelling mistakes",
    "grammar": "1 = no sentence structure, 3 = sentences exist but are often broken, 5 = fluent and grammatical",
    "consistency": "1 = no stable characters or facts, 3 = names, roles or facts change partway through, "
                   "5 = characters and facts stay consistent",
    "prompt_following": "1 = ignores the opening text, 3 = loosely related to it, 5 = a natural continuation of it",
    "structure": "1 = no story, 3 = some events but no clear arc, 5 = clear setup, events and resolution",
    "meaning": "1 = nonsense, 3 = individual sentences make sense but the whole does not, "
               "5 = makes sense with clear cause and effect",
    "style": "1 = nothing like a children's story, 3 = partly, 5 = simple words, short sentences, "
             "suitable for a 3-4 year old, like a typical TinyStories story",
}

SYSTEM_PROMPT = """You are grading short stories written by small language models that were trained on \
TinyStories, a dataset of simple children's stories. Each story starts with a given opening text and may \
be cut off at a length limit, so do not penalise a story only for stopping mid-sentence.

Score each criterion from 1 to 5:
""" + "\n".join(f"- {name}: {desc}" for name, desc in CRITERIA.items()) + """

Be strict and use the whole scale. First write a short justification, then give the scores."""

SCHEMA = {
    "type": "object",
    "properties": {
        "justification": {"type": "string"},
        **{name: {"type": "integer", "minimum": 1, "maximum": 5} for name in CRITERIA},
    },
    "required": ["justification", *CRITERIA],
}


class QuotaExhausted(Exception):
    pass


class Judge:

    def __init__(self, model, provider="gemini", rpm=10, max_retries=6):
        self.model = model
        self.base_url, key_env = PROVIDERS[provider]
        self.api_key = os.environ.get(key_env)
        if not self.api_key:
            raise RuntimeError(f"set the {key_env} environment variable")
        self.min_interval = 60 / rpm
        self.max_retries = max_retries
        self._last_call = 0.0

    def _post(self, body):
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                message = e.read().decode(errors="ignore")
                if e.code != 429 and e.code < 500:
                    raise RuntimeError(f"{e.code}: {message}") from e
                # a daily quota says "retry in 15h..." and there's no point waiting for it
                if e.code == 429 and re.search(r"retry in \d+h", message):
                    raise QuotaExhausted(message.split("Please retry")[0].strip()) from e
                delay = 10 * 2 ** attempt
                print(f"  got {e.code}, retrying in {delay}s")
                time.sleep(delay)
        raise RuntimeError("judge failed after retries")

    def score(self, prompt, story):
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Opening text: {prompt!r}\n\nStory:\n{story}"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "story_scores", "schema": SCHEMA, "strict": True},
            },
        }
        # every so often the reply comes back empty or cut off, so just ask again
        for _ in range(3):
            response = self._post(body)
            try:
                result = json.loads(response["choices"][0]["message"]["content"])
                for name in CRITERIA:
                    result[name] = min(5, max(1, int(result[name])))
                return result
            except (KeyError, IndexError, TypeError, ValueError):
                print(f"  bad reply, retrying: {json.dumps(response)[:300]}")
        raise RuntimeError("judge kept returning unusable replies")
