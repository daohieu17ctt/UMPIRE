import pickle
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--generation_dir", type=str, required=True,
                    help='Directory containing generation files to merge')
args = parser.parse_args()

outdir = args.generation_dir
objs = []
dir_list = sorted(os.listdir(outdir), key=lambda s: int(s.split("_")[-1]))
for split in dir_list:
    filepath = os.path.join(outdir, split)
    filename = os.listdir(filepath)[0]
    filepath = os.path.join(filepath, filename)
    if os.path.isfile(filepath):
        tmp_obj = pickle.load(open(filepath, 'rb'))
        objs.extend(tmp_obj)
        print(split, len(tmp_obj))

with open(f"{outdir}.pkl", 'wb') as fwriter:
    pickle.dump(objs, fwriter)

print(len(objs))
with open(f"{outdir}.pkl", 'rb') as reader:
    print(len(pickle.load(reader)))
