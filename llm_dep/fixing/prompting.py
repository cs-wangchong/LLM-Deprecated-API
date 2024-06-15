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

from llm_dep.models import *
from llm_dep.utils.metric_utils import Bleu, CodeBleu
from llm_dep.utils.log_utils import init_log
from llm_dep.utils.source_utils import *



PROMPT_DICT = {
    # "p1": "{indent}# {dep_stmt}\n{indent}# Use {rep} instead of {dep} and refine the arguments.",  # best so far
    # "p2": "{indent}# {dep_stmt}\n{indent}# fix: {dep} is deprecated, use {rep} instead and revise the return value and arguments.",
    # "p3": "{indent}# Fix: {dep} is deprecated, use {rep} instead.\n{indent}# {dep_stmt}",
    # "p4": "{indent}# {dep} is deprecated, use {rep} instead.\n{indent}# {dep_stmt}",
    # "p5": "{indent}# {dep} -> {rep}, revise the return value and arguments.\n{indent}# {dep_stmt}",
    # "p6": "-{indent}{dep_stmt}\n+{indent}# {dep}(...) is deprecated, use {rep}(...) instead.\n+",
    # "p7": "{indent}# {dep} -> {rep}\n-{indent}{dep_stmt}\n+",
    "p8": "{indent}# {dep_stmt}\n{indent}# {dep} is deprecated, use {rep} instead and revise the return value and arguments.",
    # "p9": "{indent}# {dep} is deprecated, use {rep} instead and revise the return value and arguments.\n-{indent}{dep_stmt}\n+",
    # "p10": "{indent}# {dep_stmt}\n{indent}# {dep} is deprecated, use {rep} instead and correct the return value and arguments.",
}


def preprocess(item):
    lines = item["probing input"].split("\n")
    pred, api_preds = item["probing predictions"][0]
    api_preds = list(set(item['deprecated api']) & set(api_preds))
    line_prefix = ""

    pred_lines = pred.split("\n")
    if pred_lines[0].strip() == "":
        while len(pred_lines) > 0 and pred_lines[0].strip() == "":
            pred_lines.pop(0)
        pred = "\n".join(pred_lines)
    probing_input = "\n".join(lines).rstrip()

    indent = re.search("^\s*", pred.split("\n")[0]).group(0)
    dep_api = item["alias dict"].get(api_preds[0], api_preds[0])
    rep_api = item["alias dict"].get(item["replacement api"], item["replacement api"])
    dep_stmt = extract_first_statement(pred, remove_space=False)
    # dep_stmt = dep_stmt.replace(dep_api, rep_api)
    prompt = PROMPT_TEMPLATE.format(dep=dep_api, rep=rep_api, dep_stmt=dep_stmt, indent=indent)
    prompting_input = f"{probing_input}\n\n{prompt}\n{line_prefix}"
    item["prompting input"] = prompting_input.rstrip()
    return item



parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name', required=True, type=str)
parser.add_argument('--lib', help='library name', required=True, type=str)
parser.add_argument('--prompt', help='prompt template id', required=False, type=str, default="p8")
parser.add_argument('--nodepstmt', help='without deprecated statement', action='store_true')
parser.add_argument('--maxlen', help='max length', required=False, type=int, default=50)
parser.add_argument('--beam', help='beam size', required=False, type=int, default=1)
parser.add_argument('--batch', help='batch size', required=False, type=int, default=4)


if __name__ == "__main__":
    args = parser.parse_args()

    MODEL = args.model
    LIB = args.lib
    PROMPT_TEMPLATE = PROMPT_DICT[args.prompt]
    WITHOUT_DEP_STMT = args.nodepstmt
    MAX_LEN = args.maxlen
    BEAM = args.beam
    BATCH_SIZE = args.batch
    
    PROBING_RESULTS = f"probing-results/{LIB}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.json"

    OUTPUT_DIR = f"fixing-results-prompting/{LIB}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    init_log(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-{args.prompt}{'-nodep' if WITHOUT_DEP_STMT else '-withdep'}-maxlen{MAX_LEN}-beam{BEAM}.log")


    with Path(PROBING_RESULTS).open("r") as f:
        samples = json.load(f)

    test_set = []
    for item in samples:
        # if item["category"] != "outdated" or item["source"] != "github-commit":
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
    for beg, end in tqdm(list(zip(range(0, len(test_set), BATCH_SIZE), range(BATCH_SIZE, len(test_set) + BATCH_SIZE, BATCH_SIZE)))):
        batch = test_set[beg:end]
        inputs = [item["prompting input"] for item in batch]
        preds = engine.complete(
            inputs,
            max_len=MAX_LEN,
            beam_size=BEAM,
            cand_num=1
        )
    
        for item, _preds in zip(batch, preds):
            _preds = [clean_pred(p) for p in _preds]
            # _preds = [extract_first_func(item["prompting input"] + p)[len(extract_first_func(item["prompting input"])):] for p in _preds]
            _api_preds = [extract_apis_in_first_stmt(p, item["reference dict"], item["alias dict"]) for p in _preds]
            
            item["prompting predictions"] = list(zip(_preds, _api_preds))

            probing_preds, _ = zip(*item["probing predictions"])
            
            logging.info("#" * 40)
            logging.info(f"api mapping: {item['deprecated api']} -> {item['replacement api']}")
            logging.info("")
            logging.info("")
            logging.info(f"function:\n{item['function']}")
            logging.info(f"##### before prompting ######")
            logging.info(f"probing input:\n{item['probing input']}")
            logging.info(f"predictions:\n{probing_preds}")
            logging.info(f"##### after prompting ######")
            logging.info(f"prompting input:\n{item['prompting input']}")
            logging.info(f"predictions:\n{_preds}")
            logging.info(f"stmt predictions:\n{[extract_first_statement(p, False) for p in _preds]}")
            logging.info(f"api prediction:\n{_api_preds}")

            prediction_info[f"{item['category']} total"] += 1

            _apis = set()
            for s in _api_preds:
                _apis.update(s)
            if len(set(item['deprecated api']) & _apis) > 0:
                logging.info(f"OH NO! The model predicts the deprecated API `{set(item['deprecated api']) & _apis}`!")
                prediction_info[f"{item['category']} llm-bad"] += 1
            elif item['replacement api'] in _apis:
                logging.info(f"WOW! The model predicts the replacement API `{item['replacement api']}`!")
                prediction_info[f"{item['category']} llm-good"] += 1
            else:
                prediction_info[f"{item['category']} llm-other"] += 1


            logging.info("")
            logging.info("")
            logging.info("")
            logging.info("")

        results.extend(batch)

    with Path(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-{args.prompt}{'-nodep' if WITHOUT_DEP_STMT else '-withdep'}-maxlen{MAX_LEN}-beam{BEAM}.json").open("w") as f:
        json.dump(results, f, indent=4)

    DMP = diff_match_patch()
    references, probing_predictions, prompting_predictions = [], [], []
    for item in results:
        reference = item["reference"]
        # reference = normalize_stmt(reference)
        references.append(reference)
        probing_prediction = item["probing predictions"][0][0]
        probing_prediction = extract_first_statement(probing_prediction)
        # probing_prediction = normalize_stmt(probing_prediction)
        probing_predictions.append(probing_prediction)
        prompting_prediction = item["prompting predictions"][0][0]
        prompting_prediction = extract_first_statement(prompting_prediction)
        # prompting_prediction = normalize_stmt(prompting_prediction)
        prompting_predictions.append(prompting_prediction)
        logging.info("#" * 50)
        logging.info(f"reference: {reference}")
        logging.info(f"probing prediction: {probing_prediction}")
        logging.info(f"prompting prediction: {prompting_prediction}")
        if prompting_prediction == reference:
            logging.info("\tEXACT MATCH!")
        else:
            diffs = DMP.diff_main(reference, prompting_prediction)
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
    prompting_prediction_tokens = [tokenizer.tokenize(pred) for pred in prompting_predictions]
    logging.info("\n\n\n\n")
    logging.info("########### BEFORE PROMPTING ##########")
    logging.info(f"INSTANCE NUMBER: {len(references)}")
    logging.info(f"\tBLEU-4: {Bleu.compute_bleu(reference_tokens, probing_prediction_tokens, smooth=True) if len(references) > 0 else 0}")
    logging.info(f"\tCODEBLEU-4: {CodeBleu.compute_codebleu(references, probing_predictions) if len(references) > 0 else 0}")
    logging.info(f"\tEDIT SIMILARITY: {sum([fuzz.ratio(' '.join(ref.split()), ' '.join(pred.split())) for ref, pred in zip(references, probing_predictions)]) / len(references) if len(references) > 0 else 0}")
    logging.info(f"\tEXACT MATCH: {sum([(1 if ref == pred else 0) for ref, pred in zip(references, probing_predictions)]) / len(references) if len(references) > 0 else 0}")

    logging.info("\n\n\n\n")
    logging.info("########### AFTER PROMPTING ##########")
    logging.info(f"INSTANCE NUMBER: {len(references)}")
    logging.info(f"\tBLEU-4: {Bleu.compute_bleu(reference_tokens, prompting_prediction_tokens, smooth=True) if len(references) > 0 else 0}")
    logging.info(f"\tCODEBLEU-4: {CodeBleu.compute_codebleu(references, prompting_predictions) if len(references) > 0 else 0}")
    logging.info(f"\tEDIT SIMILARITY: {sum([fuzz.ratio(' '.join(ref.split()), ' '.join(pred.split())) for ref, pred in zip(references, prompting_predictions)]) / len(references) if len(references) > 0 else 0}")
    logging.info(f"\tEXACT MATCH: {sum([(1 if ref == pred else 0) for ref, pred in zip(references, prompting_predictions)]) / len(references) if len(references) > 0 else 0}")

    logging.info("\n\n\n\n")
    logging.info("##### API PROBING STATISTICS #####")
    logging.info(f"\n{json.dumps(prediction_info, indent=4)}")