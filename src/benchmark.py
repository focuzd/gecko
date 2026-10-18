import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict

import torch

from data import CharTokenizer, load_text
from judge import CRITERIA, Judge, QuotaExhausted
from sample_stories import END_MARKER, generate_story
from utils import OUTPUT_DIR, ROOT, WEIGHTS_DIR, get_device, load_checkpoint

BENCH_DIR = OUTPUT_DIR / "benchmark"
RESULT_FILE = ROOT / "result.md"
STORIES_FILE = BENCH_DIR / "stories.jsonl"

MODELS = ["bigram", "mlp", "gpt"]

PROMPTS = {
    "in": ["Once upon a time", "One day, a little girl named Lily"],
    "near": ["Once upon a time, a robot named Zed", "One day, a grumpy dragon wanted to"],
    "out": ["The scientist looked at the data and", "In the year 3000, the city"],
}

STORIES_PER_PROMPT = 4
MAX_NEW_CHARS = 600
TEMPERATURE = 0.8
N_REFERENCE = 12
N_RANDOM = 6

# meaning, consistency and structure are what separate a story from fluent-looking text
WEIGHTS = {name: 1.0 for name in CRITERIA}
WEIGHTS.update(meaning=1.5, consistency=1.5, structure=1.5)


def all_prompts():
    return [(group, prompt) for group, prompts in PROMPTS.items() for prompt in prompts]


def train_split_text():
    text = load_text("tinystories")
    n = int(0.9 * len(text))
    return text[:n], text[n:]


def generate():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    stories = []

    for model_name in MODELS:
        model, ckpt = load_checkpoint(WEIGHTS_DIR / f"tinystories_{model_name}.pt", device)
        tok = CharTokenizer(ckpt["chars"])
        for p, (group, prompt) in enumerate(all_prompts()):
            dropped = set(prompt) - set(tok.chars)
            if dropped:
                print(f"warning: {prompt!r} has characters the model doesn't know: {dropped}")
            for k in range(STORIES_PER_PROMPT):
                torch.manual_seed(1000 * p + k)  # same seeds for every model
                text, ended = generate_story(model, tok, ckpt["block_size"], prompt,
                                             MAX_NEW_CHARS, TEMPERATURE)
                stories.append(dict(id=f"{model_name}-{p}-{k}", source=model_name, group=group,
                                    prompt=prompt, text=text, ended=ended))
            print(f"{model_name}: done with {prompt!r}")

    # real TinyStories stories from the held out split, cut to the same length
    _, val_text = train_split_text()
    real = [s.strip() for s in val_text.split(END_MARKER) if len(s.strip()) > 200]
    rng = random.Random(0)
    for i, story in enumerate(rng.sample(real[1:-1], N_REFERENCE)):  # first/last may be partial
        prompt = " ".join(story.split()[:4])
        stories.append(dict(id=f"reference-{i}", source="reference", group="reference",
                            prompt=prompt, text=story[:len(prompt) + MAX_NEW_CHARS],
                            ended=len(story) <= len(prompt) + MAX_NEW_CHARS))

    # random characters as the floor
    chars = sorted(set(val_text) - {"\n"})
    for i in range(N_RANDOM):
        group, prompt = all_prompts()[i % len(all_prompts())]
        noise = "".join(rng.choice(chars) for _ in range(MAX_NEW_CHARS))
        stories.append(dict(id=f"random-{i}", source="random", group="random",
                            prompt=prompt, text=prompt + noise, ended=False))

    with open(STORIES_FILE, "w", encoding="utf-8") as f:
        for s in stories:
            f.write(json.dumps(s) + "\n")
    print(f"saved {len(stories)} stories to {STORIES_FILE}")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judgments_file(judge_model):
    return BENCH_DIR / f"judgments_{judge_model}.jsonl"


def judge(judge_model, provider, rpm):
    stories = load_jsonl(STORIES_FILE)
    out = judgments_file(judge_model)
    done = {j["id"] for j in load_jsonl(out)} if out.exists() else set()

    # shuffled so the judge never sees one model's stories in a row
    todo = [s for s in stories if s["id"] not in done]
    random.Random(1).shuffle(todo)
    print(f"{len(done)} already judged, {len(todo)} to go")

    j = Judge(judge_model, provider, rpm)
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            try:
                result = j.score(s["prompt"], s["text"])
            except QuotaExhausted as e:
                print(f"daily quota used up, stopping. rerun later to continue.\n{e}")
                return
            except RuntimeError as e:
                print(f"[{i}/{len(todo)}] {s['id']}: skipped ({e}), rerun to retry")
                continue
            f.write(json.dumps({"id": s["id"], **result}) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {s['id']}: " + " ".join(str(result[c]) for c in CRITERIA))


def weighted_score(j):
    return sum(WEIGHTS[c] * j[c] for c in CRITERIA) / sum(WEIGHTS.values())


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def auto_metrics(story, known_words, train_ngrams, n=8):
    w = words(story["text"][len(story["prompt"]):])
    bigrams = list(zip(w, w[1:]))

    # longest run of consecutive n-grams that appear verbatim in the training data
    longest = run = 0
    for i in range(len(w) - n + 1):
        run = run + 1 if tuple(w[i:i + n]) in train_ngrams else 0
        longest = max(longest, run)

    return {
        "real_words": sum(x in known_words for x in w) / max(len(w), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "ended": float(story["ended"]),
        "longest_copy": longest + n - 1 if longest else 0,
    }


def mean_std(xs):
    if not xs:
        return "-"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):.2f} ± {sd:.2f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def report():
    stories = {s["id"]: s for s in load_jsonl(STORIES_FILE)}
    sources = MODELS + ["reference", "random"]

    train_text, _ = train_split_text()
    train_words = words(train_text)
    known_words = {w for w, c in Counter(train_words).items() if c >= 3}
    train_ngrams = {tuple(train_words[i:i + 8]) for i in range(len(train_words) - 7)}
    metrics = {sid: auto_metrics(s, known_words, train_ngrams) for sid, s in stories.items()}

    sections = ["# Benchmark results\n",
                f"{STORIES_PER_PROMPT} stories per prompt, {len(all_prompts())} prompts, "
                f"max {MAX_NEW_CHARS} new characters, temperature {TEMPERATURE}.\n"]

    by_source = defaultdict(list)
    for sid, m in metrics.items():
        by_source[stories[sid]["source"]].append(m)
    rows = [[src] + [mean_std([m[k] for m in by_source[src]]) for k in
                     ["real_words", "distinct_2", "ended", "longest_copy"]] for src in sources]
    sections.append("## Automatic metrics\n\n" + table(
        ["source", "real word rate", "distinct-2", "ended on its own", "longest copied span (words)"], rows))

    judge_files = sorted(BENCH_DIR.glob("judgments_*.jsonl"))
    all_judgments = {}
    for path in judge_files:
        name = path.stem.removeprefix("judgments_")
        judgments = {j["id"]: j for j in load_jsonl(path) if j["id"] in stories}
        if len(judgments) < len(stories):
            sections.append(f"## Judge: {name}\n\nIncomplete ({len(judgments)}/{len(stories)} stories), left out.")
            continue
        all_judgments[name] = judgments
        for j in judgments.values():
            j["weighted"] = weighted_score(j)

        groups = defaultdict(list)
        for sid, j in judgments.items():
            groups[stories[sid]["source"]].append(j)

        floor = statistics.mean(j["weighted"] for j in groups["random"])
        ceiling = statistics.mean(j["weighted"] for j in groups["reference"])

        def normalised(js):
            return 100 * (statistics.mean(j["weighted"] for j in js) - floor) / (ceiling - floor)

        rows = [[src, len(groups[src])] + [mean_std([j[c] for j in groups[src]]) for c in CRITERIA]
                + [mean_std([j["weighted"] for j in groups[src]]), str(round(normalised(groups[src])))]
                for src in sources if groups[src]]
        sections.append(f"## Judge: {name}\n\nScores 1-5. The normalised score puts random characters "
                        f"at 0 and real TinyStories at 100.\n\n" + table(
                            ["source", "n", *CRITERIA, "weighted", "normalised"], rows))

        rows = []
        for src in MODELS:
            row = [src]
            for g in PROMPTS:
                js = [j for j in groups[src] if stories[j["id"]]["group"] == g]
                row.append(str(round(normalised(js))) if js else "-")
            rows.append(row)
        sections.append(f"### Normalised score by prompt type ({name})\n\n" + table(
            ["model", "in-distribution", "near", "out-of-distribution"], rows))

    if len(all_judgments) >= 2:
        names = list(all_judgments)
        a, b = all_judgments[names[0]], all_judgments[names[1]]
        common = a.keys() & b.keys()
        rows = []
        for c in [*CRITERIA, "weighted"]:
            diffs = [abs(a[i][c] - b[i][c]) for i in common]
            within = sum(d <= 1 for d in diffs) / len(diffs)
            rows.append([c, f"{statistics.mean(diffs):.2f}", f"{100 * within:.0f}%"])
        sections.append(f"## Judge agreement: {names[0]} vs {names[1]} ({len(common)} stories)\n\n" + table(
            ["criterion", "mean abs difference", "within 1 point"], rows))

    summary = "\n\n".join(sections) + "\n"
    RESULT_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"saved {RESULT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["generate", "judge", "report"])
    parser.add_argument("--judge-model", default="gemini-3.8-flash")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openrouter"])
    parser.add_argument("--rpm", type=float, default=10, help="max judge requests per minute")
    args = parser.parse_args()

    if args.step == "generate":
        generate()
    elif args.step == "judge":
        judge(args.judge_model, args.provider, args.rpm)
    else:
        report()
