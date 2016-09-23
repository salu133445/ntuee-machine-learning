from PIL import Image
import numpy as np
import argparse

outputFile = 'ans2.png'

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()

tmp = np.asarray(Image.open(args.image))
tmp1 = np.rot90(tmp, 2)
result = Image.fromarray(tmp1)
result.save(outputFile)
