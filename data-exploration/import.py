import glob
import subprocess
from tqdm import tqdm
import pickle


files = glob.glob("results/*.txt")
results = []
for file in tqdm(files):
    results.append(subprocess.run([
        'sqlite3',
        'object_detection.db',
        '-cmd',
        '.mode csv',
        f'.import {file} boxes'
    ], capture_output=True))


with open("import_statuses.pickle", "wb") as f:
    pickle.dump(results, f)
