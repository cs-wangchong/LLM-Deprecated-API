import sys
import os
import math
import json
import random
import time
from collections import defaultdict
from tqdm import tqdm
import re
from pathlib import Path
from multiprocessing import Pool
import ast
import traceback


from api_matcher import APIMatcher

def handle_error(exception):
    traceback.print_exc()


def process_one(filename):
    try:
        code = open(filename).read()
        results = APIMatcher(code, apis, maxline=1000).match().matched_funcs
    except Exception:
        traceback.print_exc()
        return None
    json_data = {
        "filename": filename,
        "source": code
    }
    
    if len(results) == 0:
        return None

    json_data["matching"] = results
    return json_data


def process_batch(batch_id, batch, output_path):
    out = Path(output_path).open("w")
    for filename in tqdm(batch, desc=f"batch-{batch_id}", ascii=True):
        json_data = process_one(filename)
        if json_data is None:
            continue
        out.write(f"{json.dumps(json_data)}\n")
        out.flush()
    out.close()


LIBs = [
    "tensorflow",
    "pytorch",
    # "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "seaborn",
    "transformers"
]

N_THREADS = 12


if __name__ == '__main__':
    file_count = 0
    for lib in LIBs:
        MAPPINGS_FILE = f"data/mappings/deprecated-mappings-{lib}.json"

        SOURCE_DIR = f"data/searching-results/{lib}/sourcegraph"
        OUTPUT_DIR = f"data/matching-results/{lib}/sourcegraph"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        with Path(MAPPINGS_FILE).open("r") as f:
            mappings = json.load(f)
        for dep, rep in list(mappings.items()):
            mappings[dep.replace("tensorflow.compat.v1.", "tensorflow.")] = rep.replace("tensorflow.compat.v1.", "tensorflow.")
        apis = set()
        for dep, rep in list(mappings.items()):
            apis.add(dep)
            apis.add(rep)
        
        # filenames = []
        # for json_path in Path(f"{SOURCE_DIR}").glob("*.json"):
        #     print(json_path)
        #     with json_path.open("r") as f:
        #         items = json.load(f)
        #     for item in items:
        #         repo = item["repository"][len("github.com/"):]
        #         commit = item["commit"]
        #         file = item["path"]
        #         source_path = Path(f"{SOURCE_DIR}/{repo.replace('/', '#')}-{commit}/{file}")

        #         if not source_path.exists():
        #             continue
        #         filenames.append(str(source_path))

        filenames = [str(py_path) for py_path in Path(SOURCE_DIR).rglob("*.py")]

        random.shuffle(filenames)
        file_count += len(filenames)

        # process_batch(1, filenames[:50], "tmp.jsonl")
        
        # pool = Pool(N_THREADS)
        
        # batch_size = math.ceil(len(filenames) / N_THREADS)
        # ranges = list(zip(range(0, len(filenames), batch_size), range(batch_size, len(filenames) + batch_size, batch_size)))
        # for idx, (beg, end) in enumerate(ranges):
        #     batch = filenames[beg:end]
        #     # process_batch(idx+1, batch, f"{OUTPUT_DIR}/matching-batch-{idx+1}.jsonl")
        #     pool.apply_async(process_batch, (idx+1, batch, f"{OUTPUT_DIR}/matching-batch-{idx+1}.jsonl"), error_callback=handle_error)
        # pool.close()
        # pool.join()

        # samples = []
        # visited_funcs = set()
        # api2count = defaultdict(int)
        # for batch_idx in range(1, N_THREADS):
        #     with Path(f"{OUTPUT_DIR}/matching-batch-{batch_idx}.jsonl").open("r") as f:
        #         for line in f:
        #             line = line.strip()
        #             if line == "":
        #                 continue
        #             json_data = json.loads(line)
        #             if len(json_data["matching"]) == 0:
        #                 continue
        #             for item in json_data["matching"]:
        #                 if item["function"] in visited_funcs:
        #                     continue
        #                 visited_funcs.add(item["function"])
        #                 samples.append(item)
        #                 api = item["matched call"]["matched api"]
        #                 api2count[api] += 1
        # api2count = dict(sorted(api2count.items(), key=lambda x: x[1], reverse=True))
        # print(f"{lib}: {len(samples)}")

        # with Path(f"{OUTPUT_DIR}/matched-functions.json").open("w") as f:
        #     json.dump(samples, f, indent=4)
        # with Path(f"{OUTPUT_DIR}/matched-apis.json").open("w") as f:
        #     json.dump(dict(api2count), f, indent=4)

    print(f"total file: {file_count}")