import sys
import json
import logging
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import traceback
import argparse
import re
from fuzzywuzzy import fuzz

from llm_dep.models import MODEL_FACTORY, CompletionEngine
from llm_dep.utils.metric_utils import Bleu, CodeBleu
from llm_dep.utils.log_utils import init_log
from llm_dep.utils.source_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name', required=True, type=str)
parser.add_argument('--lib', help='library name', required=True, type=str)
parser.add_argument('--maxlen', help='max length', required=False, type=int, default=50)
parser.add_argument('--beam', help='beam size', required=False, type=int, default=1)
parser.add_argument('--batch', help='batch size', required=False, type=int, default=4)


if __name__ == "__main__":
    args = parser.parse_args()

    MODEL = args.model
    LIB = args.lib
    MAX_LEN = args.maxlen
    BEAM = args.beam
    BATCH_SIZE = args.batch


    SAMPLES_FILE = f"probing-inputs/{LIB}/samples.json"
    OUTPUT_DIR = f"probing-results/{LIB}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    init_log(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.log")

    with Path(SAMPLES_FILE).open("r") as f:
        samples = json.load(f)

    print(len(samples))

    model, tokenizer = MODEL_FACTORY[MODEL]()
    engine = CompletionEngine(model, tokenizer)

        
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

    for beg, end in tqdm(list(zip(range(0, len(samples), BATCH_SIZE), range(BATCH_SIZE, len(samples) + BATCH_SIZE, BATCH_SIZE)))):
        batch = samples[beg:end]
        inputs = [item["probing input"] for item in batch]
        
        preds = engine.complete(
            inputs,
            max_len=MAX_LEN,
            beam_size=BEAM,
            cand_num=BEAM
        )
    
        for item, _preds in zip(batch, preds):
            _preds = [clean_pred(p) for p in _preds]
            _preds = [extract_first_func(item["probing input"] + p)[len(item["probing input"]):] for p in _preds]
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

            logging.info("")
            logging.info("")
            logging.info("")
            logging.info("")

        results.extend(batch)

    with Path(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.json").open("w") as f:
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