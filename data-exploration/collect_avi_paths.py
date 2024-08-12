import time
import glob
import os
import multiprocessing


OUT_FOLDER = "tmp"
os.makedirs(OUT_FOLDER, exist_ok=True)


def collect_paths(disk_name: str) -> tuple[str, str]:
    with open(os.path.join(OUT_FOLDER, f"{disk_name}-files.txt"), "w") as f:
        for file in glob.iglob(os.path.join("/mnt", disk_name, "**/*.avi.seq"), recursive=True):
            f.write(f"{file}\n")

    return disk_name, "OK"

t0 = time.time()
with multiprocessing.Pool(2) as p:
    print(p.map(collect_paths, ["pg-fs01","pg-fs02"]))
t1= time.time()

t = t1 - t0
print(f"Took {t // 60 }m {int(t % 60)}s")
