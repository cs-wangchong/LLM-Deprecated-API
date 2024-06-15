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
from diff_match_patch import diff_match_patch

from llm_dep.models import MODEL_FACTORY, CompletionEngine
from llm_dep.utils.metric_utils import Bleu, CodeBleu
from llm_dep.utils.log_utils import init_log
from llm_dep.utils.source_utils import *



def preprocess(item):
    _pred, _api_preds = item["probing predictions"][0]
    target_apis = set(item['deprecated api']) & set(_api_preds)
    idx = index_of_api(_pred, target_apis, item["reference dict"], item["alias dict"])
    item["replacing input"] = item["probing input"] + _pred[:idx] + item["expected call"]
    return item


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
    
    PROBING_RESULTS = f"probing-results/{LIB}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.json"

    OUTPUT_DIR = f"fixing-results-replacing/{LIB}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    init_log(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.log")


    with Path(PROBING_RESULTS).open("r") as f:
        samples = json.load(f)

    test_set = []
    for item in samples:
        # if item["category"] != "buggy" or item["source"] != "github-commit":
        #     continue
        if item["category"] != "up-to-dated":
            continue
        _, _api_preds = item["probing predictions"][0]
        if len(set(item['deprecated api']) & set(_api_preds)) == 0:
            continue
        test_set.append(preprocess(item))
        

    model, tokenizer = MODEL_FACTORY[MODEL]()
    engine = CompletionEngine(model, tokenizer)


    results = []
    for beg, end in tqdm(list(zip(range(0, len(test_set), BATCH_SIZE), range(BATCH_SIZE, len(test_set) + BATCH_SIZE, BATCH_SIZE)))):
        batch = test_set[beg:end]
        inputs = [item["replacing input"] for item in batch]
        preds = engine.complete(
            inputs,
            max_len=MAX_LEN,
            beam_size=BEAM,
            cand_num=1
        )
    
        for item, _preds in zip(batch, preds):
            _preds = [clean_pred(p) for p in _preds]
            _preds = [extract_first_func(item["replacing input"] + p)[len(item["probing input"]):] for p in _preds]
            _api_preds = [extract_apis_in_first_stmt(p, item["reference dict"], item["alias dict"]) for p in _preds]
            
            item["replacing predictions"] = list(zip(_preds, _api_preds))

            probing_preds, probing_api_preds = zip(*item["probing predictions"])
            
            logging.info("#" * 40)
            logging.info(f"api mapping: {item['deprecated api']} -> {item['replacement api']}")
            logging.info("")
            logging.info("")
            logging.info(f"function:\n{item['function']}")
            logging.info(f"##### before replacing ######")
            logging.info(f"probing input:\n{item['probing input']}")
            logging.info(f"predictions:\n{probing_preds}")
            logging.info(f"stmt predictions:\n{[extract_first_statement(p, False) for p in probing_preds]}")
            logging.info(f"api predictions:\n{probing_api_preds}")
            logging.info(f"##### after replacing ######")
            logging.info(f"replacing input:\n{item['replacing input']}")
            logging.info(f"predictions:\n{_preds}")
            logging.info(f"stmt predictions:\n{[extract_first_statement(p, False) for p in _preds]}")
            logging.info(f"api predictions:\n{_api_preds}")


            logging.info("")
            logging.info("")
            logging.info("")
            logging.info("")

        results.extend(batch)

    with Path(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.json").open("w") as f:
        json.dump(results, f, indent=4)

    DMP = diff_match_patch()
    references, probing_predictions, replacing_predictions = [], [], []
    for item in results:
        reference = item["reference"]
        # reference = normalize_stmt(reference)
        references.append(reference)

        probing_prediction = item["probing predictions"][0][0]
        probing_prediction = extract_first_statement(probing_prediction)
        # probing_prediction = normalize_stmt(probing_prediction)
        probing_predictions.append(probing_prediction)
        replacing_prediction = item["replacing predictions"][0][0]
        replacing_prediction = extract_first_statement(replacing_prediction)
        # replacing_prediction = normalize_stmt(replacing_prediction)
        replacing_predictions.append(replacing_prediction)
        logging.info("#" * 50)
        logging.info(f"reference: {reference}")
        logging.info(f"probing prediction: {probing_prediction}")
        logging.info(f"replacing prediction: {replacing_prediction}")
        if replacing_prediction == reference:
            logging.info("\tEXACT MATCH!")
        else:
            diffs = DMP.diff_main(reference, replacing_prediction)
            DMP.diff_cleanupSemantic(diffs)
            diff_str = "\tDIFF: "
            for diff_type, diff_text in diffs:
                if diff_type == diff_match_patch.DIFF_INSERT:
                    diff_str += f"<INSERT>{diff_text}</INSERT>"
                elif diff_type == diff_match_patch.DIFF_DELETE:
                    diff_str += f"<DELETE>{diff_text}</DELETE>"
                elif diff_type == diff_match_patch.DIFF_EQUAL:
                    diff_str += f"{diff_text}"
            logging.info(diff_str)

    reference_tokens = [tokenizer.tokenize(ref) for ref in references]
    probing_prediction_tokens = [tokenizer.tokenize(pred) for pred in probing_predictions]
    replacing_prediction_tokens = [tokenizer.tokenize(pred) for pred in replacing_predictions]

    logging.info("\n\n\n\n")
    logging.info("########### BEFORE REPLACING ##########")
    logging.info(f"INSTANCE NUMBER: {len(references)}")
    logging.info(f"\tBLEU-4: {Bleu.compute_bleu(reference_tokens, probing_prediction_tokens, smooth=True) if len(references) > 0 else 0}")
    logging.info(f"\tCODEBLEU-4: {CodeBleu.compute_codebleu(references, probing_predictions) if len(references) > 0 else 0}")
    logging.info(f"\tEDIT SIMILARITY: {(sum([fuzz.ratio(' '.join(ref.split()), ' '.join(pred.split())) for ref, pred in zip(references, probing_predictions)]) / len(references)) if len(references) > 0 else 0}")
    logging.info(f"\tEXACT MATCH: {(sum([(1 if ref == pred else 0) for ref, pred in zip(references, probing_predictions)]) / len(references)) if len(references) > 0 else 0}")

    logging.info("\n\n\n\n")
    logging.info("########### AFTER REPLACING ##########")
    logging.info(f"INSTANCE NUMBER: {len(references)}")
    logging.info(f"\tBLEU-4: {Bleu.compute_bleu(reference_tokens, replacing_prediction_tokens, smooth=True) if len(references) > 0 else 0}")
    logging.info(f"\tCODEBLEU-4: {CodeBleu.compute_codebleu(references, replacing_predictions) if len(references) > 0 else 0}")
    logging.info(f"\tEDIT SIMILARITY: {(sum([fuzz.ratio(' '.join(ref.split()), ' '.join(pred.split())) for ref, pred in zip(references, replacing_predictions)]) / len(references)) if len(references) > 0 else 0}")
    logging.info(f"\tEXACT MATCH: {(sum([(1 if ref == pred else 0) for ref, pred in zip(references, replacing_predictions)]) / len(references)) if len(references) > 0 else 0}")
