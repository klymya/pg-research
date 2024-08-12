import glob
import subprocess
from tqdm import tqdm
import pickle


files = glob.glob("tmp/results/*.txt")
results = []
for file in tqdm(files):
    results.append(subprocess.run([
        'sqlite3',
        'people_detection.db',
        '-cmd',
        '.mode csv',
        f'.import {file} boxes'
    ], capture_output=True))


with open("tmp/import_statuses.pickle", "wb") as f:
    pickle.dump(results, f)
