
from collections import defaultdict as dd

def list_features(input_file, output_file):
    f_cnt = dd(int)
    for line in open(input_file):
        inputs = line.strip().split()
        for t in inputs[1:]:
            f_cnt[t] += 1
    t_len = len(f_cnt.keys())
    f_set = sorted([(k, v) for k, v in f_cnt.iteritems() if v > 1], key = lambda x: x[1], reverse = True)
    print "total feature length = {}".format(t_len)
    print "filtered feature length = {}".format(len(f_set))
    f2ind = {}
    for i, k in enumerate(f_set):
        f2ind[k[0]] = i

    fout = open(output_file, 'w')
    for line in open(input_file):
        inputs = line.strip().split()
        s = inputs[0]
        for t in inputs[1:]:
            if t not in f2ind: continue
            s += " {}".format(f2ind[t])
        fout.write(s + '\n')
    fout.close()

list_features('/remote/curtis/baidu/SSL-pipeline/database/unit0_nlm_para_200/step2_gpid_cfacts/list-tok-feat/all_list_ALL.tok_feat', 'list_features.txt')
