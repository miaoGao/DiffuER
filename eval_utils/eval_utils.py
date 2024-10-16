import glob, os
import json, jsonlines
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
from tqdm import tqdm

def get_bleu(recover, reference):
    return sentence_bleu([recover.split()], reference.split(), smoothing_function=SmoothingFunction().method4,)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

def dump_mbr_result(folder, files, sample_num):
    sentenceDict = {}
    referenceDict = {}
    sourceDict = {}
    for i in range(sample_num):
        sentenceDict[i] = []
        referenceDict[i] = []
        sourceDict[i] = []

    for path in files:
        with open(path, 'r') as f:
            cnt = 0
            for row in f:
                source = json.loads(row)['source'].strip()
                reference = json.loads(row)['reference'].strip()
                recover = json.loads(row)['recover'].strip()

                sentenceDict[cnt].append(recover)
                referenceDict[cnt].append(reference)
                sourceDict[cnt].append(source)
                cnt += 1

    recovers, references, sources = [], [], []

    for k, v in tqdm(sentenceDict.items(), total=len(sentenceDict.items())):
        if len(v) == 0 or len(referenceDict[k]) == 0:
            continue

        recovers.append(selectBest(v))
        references.append(referenceDict[k][0])
        sources.append(sourceDict[k][0])

    with jsonlines.open(f'{folder}/mbr.jsonl', 'w') as f:
        data = []
        for recover, ref, src in zip(recovers, references, sources):
            data.append({
                'recover': recover,
                'reference': ref,
                'source': src
            })
        f.write_all(data)

def preprocess_for_rouge(mbr_file_path):
    output_dir = '/'.join(mbr_file_path.split('/')[:-1])

    refs = []
    preds = []
    with jsonlines.open(mbr_file_path, 'r') as f:
        for line in f:
            refs.append(line['reference'].replace("[CLS]","").replace("[SEP]","").strip())
            preds.append(line['recover'].replace("[CLS]","").replace("[SEP]","").strip())

    with open(f"{output_dir}/refs.txt", 'w') as f:
        for ref in refs:
            f.write(ref + '\n')

    with open(f"{output_dir}/preds.txt", 'w') as f:
        for pred in preds:
            f.write(pred + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=f'output/qqp/demo1', help='folder contains sampling results')
    args = parser.parse_args()

    # args.folderjsonl
    files = glob.glob(f"{args.folder}/*/*.jsonl")
    print(args.folder)

    print(files)
    print("num files:", len(files))

    if len(files) > 0:
        num_samples = len(open(files[0], 'r').readlines())
        
        dump_mbr_result(args.folder, files, num_samples)
        preprocess_for_rouge(f"{args.folder}/mbr.jsonl")


