import os
import random
from typing import List
import json
import logging
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import argparse
import re
from fuzzywuzzy import fuzz

from transformers import AutoTokenizer
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice


from llm_dep.utils.metric_utils import Bleu, CodeBleu
from llm_dep.utils.log_utils import init_log
from llm_dep.utils.source_utils import *


os.environ["OPENAI_API_KEY"] = "YOUR KEY"

CLIENT = openai.OpenAI()

PROMPT1 = "complete and output the next line for the following python function:\n\n{code}"
PROMPT2 = "complete and output the subsequent API call for the following python function:\n\n{code}"

PRICES = {
    "gpt-3.5-turbo": (0.5 / 1E6, 1.5 / 1E6),
    "gpt-4": (30 / 1E6, 60 / 1E6),
    "gpt-4-turbo": (10 / 1E6, 30 / 1E6),
    "gpt-4o": (5 / 1E6, 15 / 1E6),
}


def request_gpt(code) -> List[str]:
    global INPUT_TOKENS
    global OUTPUT_TOKENS
    prompt = PROMPT1.format(code=code) if LINE_LEVEL else PROMPT2.format(code=code)
    try:
        chat_completion: ChatCompletion = CLIENT.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0,
            max_tokens=MAX_LEN,
        )
    except openai.APIConnectionError as e:
        logging.error("The server could not be reached")
        logging.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
        return [""]
    except openai.RateLimitError as e:
        logging.error("A 429 status code was received; we should back off a bit.")
        return [""]
    except openai.APIStatusError as e:
        logging.error("Another non-200-range status code was received")
        logging.error(e.status_code)
        logging.error(e.response)
        return [""]
    logging.info(chat_completion.usage)
    INPUT_TOKENS += chat_completion.usage.prompt_tokens
    OUTPUT_TOKENS += chat_completion.usage.completion_tokens
    return [choice.message.content.replace("```python", "").strip("`") for choice in chat_completion.choices]


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name', required=False, type=str, default="gpt-3.5-turbo")
parser.add_argument('--lib', help='library name', required=True, type=str)
parser.add_argument('--linelevel', help='line level', action='store_true')
parser.add_argument('--maxlen', help='max length', required=False, type=int, default=50)


if __name__ == "__main__":
    args = parser.parse_args()

    MODEL = args.model
    LIB = args.lib
    LINE_LEVEL = args.linelevel
    MAX_LEN = args.maxlen


    SAMPLES_FILE = f"probing-inputs/{LIB}/samples.json"
    OUTPUT_DIR = f"probing-results/{LIB}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    init_log(f"{OUTPUT_DIR}/{MODEL}/predictions{'-linelevel' if LINE_LEVEL else ''}-maxlen{MAX_LEN}.log")

    with Path(SAMPLES_FILE).open("r") as f:
        samples = json.load(f)

    samples = [preprocess(item, LINE_LEVEL) for item in samples]
        
    results = []
    prediction_info = {
        "outdated total": 0,
        "outdated llm-good": 0,
        "outdated llm-bad": 0,
        "outdated llm-other": 0,
        "up-to-dated total": 0,
        "up-to-dated llm-good": 0,
        "up-to-dated llm-bad": 0,
        "up-to-dated llm-other": 0,
    }

    INPUT_TOKENS = 0
    OUTPUT_TOKENS = 0

    for item in tqdm(samples):
        _preds = request_gpt(item["probing input"])
        logging.info(_preds)
        _api_preds = [extract_apis_in_first_stmt(p, item["reference dict"], item["alias dict"]) for p in _preds]
        
        item["probing predictions"] = list(zip(_preds, _api_preds))
        
        logging.info("#" * 40)
        logging.info(f"api mapping: {item['deprecated api']} -> {item['replacement api']}")
        logging.info("")
        logging.info("")
        logging.info(f"function:\n{item['function']}")
        logging.info(f"probing input:\n{item['probing input']} ")
        logging.info(f"predictions:\n{_preds}")
        logging.info(f"stmt predictions:\n{[extract_first_statement(p, False) for p in _preds]}")
        logging.info(f"api predictions:\n{_api_preds}")

        prediction_info[f"{item['category']} total"] += 1

        _apis = set()
        for s in _api_preds:
            _apis.update(s)
        if len(set(item['deprecated api']) & _apis) > 0:
            logging.info(f"OH NO! The model predicts the deprecated API `{set(item['deprecated api']) & _apis}` for {item['category']} function!")
            prediction_info[f"{item['category']} llm-bad"] += 1
        elif item['replacement api'] in _apis:
            logging.info(f"WOW! The model predicts the replacement API `{item['replacement api']}` for {item['category']} function!")
            prediction_info[f"{item['category']} llm-good"] += 1
        else:
            prediction_info[f"{item['category']} llm-other"] += 1
        cost = round(INPUT_TOKENS * PRICES[MODEL][0] + OUTPUT_TOKENS * PRICES[MODEL][1], 8)
        logging.info(f"=== USAGE ===\ninput tokens: {INPUT_TOKENS}, output tokens: {OUTPUT_TOKENS}, cost: ${cost}")
        logging.info("")
        logging.info("")
        logging.info("")
        logging.info("")

        results.append(item)

    with Path(f"{OUTPUT_DIR}/{MODEL}/predictions{'-linelevel' if LINE_LEVEL else ''}-maxlen{MAX_LEN}.json").open("w") as f:
        json.dump(results, f, indent=4)

    references, predictions = [], []
    for item in results:
        reference = item["reference"]
        references.append(reference)
        prediction = item["probing predictions"][0][0]
        prediction = extract_first_statement(prediction)
        predictions.append(prediction)
        logging.info(f"reference: {reference}")
        logging.info(f"prediction: {prediction}")

    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf", padding_side="left")
    reference_tokens = [tokenizer.tokenize(ref) for ref in references]
    prediction_tokens = [tokenizer.tokenize(pred) for pred in predictions]

    logging.info("\n\n\n\n")
    logging.info("########### CODE COMPLETION METRICS ##########")
    logging.info(f"INSTANCE NUMBER: {len(references)}")
    logging.info(f"\tBLEU-4: {Bleu.compute_bleu(reference_tokens, prediction_tokens, smooth=True)}")
    logging.info(f"\tCODEBLEU-4: {CodeBleu.compute_codebleu(references, predictions)}")
    logging.info(f"\tEDIT SIMILARITY: {sum([fuzz.ratio(' '.join(ref.split()), ' '.join(pred.split())) for ref, pred in zip(references, predictions)]) / len(references)}")
    logging.info(f"\tEXACT MATCH: {sum([(1 if ref == pred else 0) for ref, pred in zip(references, predictions)]) / len(references)}")

    logging.info("\n\n\n\n")
    logging.info("##### API PROBING STATISTICS #####")
    logging.info(f"\n{json.dumps(prediction_info, indent=4)}")
