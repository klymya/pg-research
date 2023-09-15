import glob


with open("pg01-files.txt", "a") as f:
    for file in glob.iglob("/mnt/pg-fs01/**/*.avi*", recursive=True):
        f.write(f"{file}\n")
