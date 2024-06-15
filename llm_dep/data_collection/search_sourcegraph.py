import requests
import json
from pathlib import Path
import logging
import random
from typing import List
from tqdm import tqdm

from llm_dep.utils.log_utils import init_log


API_TOKEN = 'SOURGRAPH KEY'
SOURCEGRAPH_URL = "https://sourcegraph.com/.api/search/stream?q=context:global {keywords} count:10000 type:file lang:python &v=V3&t=keyword&sm=0&cm=t&max-line-len=5120"
GITHUB_URL = "https://github.com/{repo}/raw/{commit}/{file}"

LIBs = [
    # "tensorflow",
    # "pytorch",
    # "matplotlib",
    # "numpy",
    # "pandas",
    # "scipy",
    # "sklearn",
    # "seaborn"
    "transformers"
]

ALIAS = {
    "tensorflow": ["tf"],
    "numpy": ["np"],
    "pandas": ["pd"],
    "sklearn": ["sk"],
    "scipy": ["sc", "sp"],
    "seaborn": ["sn", "sns"],
}

MAX_COUNT = 1000

HEADERS = {
    'Accept': 'application/json',
    # If authentication is required, you may need to include an 'Authorization' header here
    'Authorization': f'token {API_TOKEN}'
}

if __name__ == '__main__':

    for lib in LIBs:
        MAPPINGS_FILE = f"data/mappings/deprecated-mappings-{lib}.json"
        OUTPUT_DIR = f"data/searching-results/{lib}/sourcegraph/"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        init_log(f"{OUTPUT_DIR}/search.log")
        
        apis = set()
        with Path(MAPPINGS_FILE).open("r") as f:
            for deprecated, replacement in json.load(f).items():
                apis.add(deprecated)
                apis.add(replacement)
        apis = list(sorted(apis))

        for api in tqdm(apis):
            if Path(f"{OUTPUT_DIR}/{api}.json").exists():
                continue
            logging.info(f"search API: {api}")

            items = []
            def search(keywords):
                url = SOURCEGRAPH_URL.format(keywords=keywords)
                logging.info(url)
                response = requests.get(url, headers=HEADERS, stream=True, timeout=12000)
                
                if response.status_code == 200:
                    lines = response.text.split('\n')
                    idx = 0
                    while idx < len(lines):
                        if lines[idx] == "event: matches":
                            item_str = lines[idx+1][6:]
                            items.extend(json.loads(item_str))
                            idx += 1
                        idx += 1
                else:
                    logging.error(f"Error: {response.status_code}, {response.text}")

            full_names = {f"{api}("}
            full_names.update(f"{api.replace(lib, alias)}(" for alias in ALIAS.get(api, []))
            keywords = " OR ".join(full_names)
            search(keywords)
            
            parts: List[str] = api.split(".")
            if len(parts) < 2:
                continue
            if parts[-1][0].isupper():
                keywords = [".".join(parts[:-1]), f"{parts[-1]}("]
            elif parts[-2][0].isupper():
                if len(parts) > 2:
                    keywords = [".".join(parts[:-2]), f"{parts[-2]}", f".{parts[-1]}("]
                else:
                    keywords = [f"{parts[-2]}", f".{parts[-1]}("]
            else:
                keywords = [".".join(parts[:-1]), f".{parts[-1]}("]
            keywords=" ".join(keywords)

            search(keywords)
            
            items = [item for item in items if "repoStars" in item]

            items.sort(key=lambda x:x["repoStars"], reverse=True)
            with Path(f"{OUTPUT_DIR}/{api}.json").open("w") as f:
                json.dump(items, f, indent=4)

            items = items[:MAX_COUNT]

            fetched_files = set()
            for item in items:
                repo = item["repository"][len("github.com/"):]
                commit = item["commit"]
                file = item["path"]
                if f"{repo}/{commit}/{file}" in fetched_files:
                    continue
                fetched_files.add(f"{repo}/{commit}/{file}")
                logging.info(f"\tfetch file: {repo}/{commit}/{file}")
                to_path = f"{OUTPUT_DIR}/sources/{repo.replace('/', '#')}-{commit}/{file}"
                Path(to_path).parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(GITHUB_URL.format(repo=repo, commit=commit, file=file))

                with Path(to_path).open("w") as f:
                    f.write(r.text)

             