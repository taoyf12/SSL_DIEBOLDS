
from collections import defaultdict as dd
import numpy as np
import random

random.seed(31415)

for i in range(3):
    print i
    fdevel = open("../seeds/{}/devel".format(i), 'w')
    feva = open("../seeds/{}/eva".format(i), 'w')
    count1 = 0
    count2 = 0
    for line in open('../seeds/all.cfacts'):
        if random.random() < 0.9:
            fdevel.write(line)
            count1 += 1
        else:
            count2 += 1
            feva.write(line)
    fdevel.close()
    feva.close()
    ratio = 0.0
    ratio = 1.0*count1/count2
    total = count1+count2
    print count1, count2, ratio, total

