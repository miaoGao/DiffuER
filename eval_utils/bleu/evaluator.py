# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys

from bleu import _bleu


def evaluate(ref_path, pred_path):
    refs = [x.strip() for x in open(ref_path, 'r', encoding='utf-8').readlines()]
    pres = [x.strip() for x in open(pred_path, 'r', encoding='utf-8').readlines()]
    
    assert len(refs) == len(pres)

    length = len(refs)
    count = 0
    for i in range(length):
        r = refs[i]
        p = pres[i]
        if r == p:
            count += 1
    
    acc = round(count/length*100, 2)
    
    bleus = []
    for order in [1, 2, 3, 4]:
        bleu_score = round(_bleu(ref_path, pred_path, max_order=order),2)
        bleus.append({
            f'BLEU-{order}': bleu_score
        })

    result = {
        'BLEUs': bleus,
        'Acc': acc 
    }

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--references', '-ref',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-pre',help="filename of the leaderboard predictions, in txt format.")
    
    args = parser.parse_args()
    result = evaluate(args.references, args.predictions)
    
    print(result)
    
if __name__ == '__main__':
    main()
