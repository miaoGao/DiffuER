### How to use

``` bash
ref_path=data/code2code/test.java-cs.txt.cs
pred_path=data/code2code/gen_.txt
python ./eval_utils/bleu/evaluator.py -ref $ref_path -pre $pred_path
```