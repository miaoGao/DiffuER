import jsonlines
import json

class Example():
    def __init__(self, id, src_tag, src_sent, tgt_sent):
        self.id = id
        self.src_tag = src_tag
        self.src_sent = src_sent
        self.tgt_sent = tgt_sent

    def __repr__(self):
        return str(self.__dict__)


def load_data_files(src_file, tgt_file):
    src_lines = open(src_file, 'r').readlines()
    tgt_lines = open(tgt_file, 'r').readlines()

    return src_lines, tgt_lines


def parse_example(id, src_seq, tgt_seq):
    src_tag = src_seq[:6].strip()
    src_sent = src_seq[6:].strip()
    tgt_sent = tgt_seq.strip()

    return Example(id, src_tag, src_sent, tgt_sent)


def convert2jsonl():
    splits = ['train', 'valid', 'test']
    dataset = 'wrting_prompt'

    for split in splits:
        src_file = f"data/{dataset}/{split}.wp_source"
        tgt_file = f"data/{dataset}/{split}.wp_target"
        output_file = f"data/{dataset}/{split}.jsonl"

        src_lines, tgt_lines = load_data_files(src_file, tgt_file)

        examples = []

        for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
            example = parse_example(i, src_line, tgt_line)
            examples.append(example.__dict__)

        with jsonlines.open(output_file, 'w') as f:
            f.write_all(examples)


def get_examples(dataset, split):
    src_file = f"/data/{dataset}/{split}.wp_source"
    tgt_file = f"/data/{dataset}/{split}.wp_target"


    src_lines, tgt_lines = load_data_files(src_file, tgt_file)

    examples = []

    for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        example = parse_example(i, src_line, tgt_line)
        examples.append(example)

    return examples


if __name__ == "__main__":
    examples = get_examples('wrting_prompt', 'valid')
    
    print(examples[:10])