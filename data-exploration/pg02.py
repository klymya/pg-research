import glob


with open("pg02-files.txt", "a") as f:
    for file in glob.iglob("/mnt/pg-fs02/**/*.avi*", recursive=True):
        f.write(f"{file}\n")
