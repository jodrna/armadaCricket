import os
from pathlib import Path

# the project root (this folder): every input and output of the process lives under it. Same file as prod_model/paths.py
# (20 Oct 2026), so scripts copied between the two projects work in either - each uses its own project's root.
PROJECT_ROOT = Path(__file__).resolve().parent

# local cache, NOT synced: only re-pullable caches of DB pulls / big round trips go here - nothing the process needs that
# can't be rebuilt by rerunning the script. Override with PROD_CACHE_DIR.
CACHE_DIR = Path(os.environ.get('PROD_CACHE_DIR', Path.home() / 'Documents' / 'Tempdata'))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
