#  CoNT: Contrastive Neural Text Generation

## you can find the whole program in https://github.com/ginkolove/AA-CoNT
## Dependencies
Main libraries
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [transformers](https://github.com/huggingface/transformers) 4.21.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 1.0.0beta
```
pip install transformers==4.21.0
pip install fastNLP==1.0.0beta
```

	
All code only supports running on Linux.


### Datasets
You can Download the jsonl files through these links.
1. Summarization：
    - [XSum](https://drive.google.com/file/d/1t--UZo4Pnv4HjGhAfun5vDz3JCoqIggq/view?usp=sharing)
    - [Multi-News](https://drive.google.com/file/d/16VdfzvLmmOrYsayujA-Hu4d3i_ejHTln/view?usp=sharing)
2. Translation：
    - [WMT'16 Ro-En](https://drive.google.com/file/d/1rGoylmZvIhNvsoPZda7OZP_0nYUfUpoq/view?usp=sharing)
    - [WMT'14 En-De and IWSLT'14 De-En](https://github.com/ChenxinAn-fdu/CoNT)
3. Code comment Generation
    - [java](https://drive.google.com/file/d/1PBdxKvMTvfCzseactMRffTUwuTI7oAGz/view?usp=sharing)
    - [python](https://drive.google.com/file/d/189xlRW3r3UuMTko73zURfJ3I_LXQ026D/view?usp=sharing)
4. Data-to-text generation  
    - [Wikibio](https://drive.google.com/file/d/1i0BZykxifH2hEdCyB_nZFvs2PT4UdUFJ/view?usp=sharing)
    - [ToTTo](https://drive.google.com/file/d/1nOlhGKpTWPCmAwmEI_gdALkAXlMn2Tbk/view?usp=sharing) (Blind test set)
5. Commonsense generation  
    - [CommonGen](https://drive.google.com/file/d/1UvCBenGMzdQyR25ka_1vmaPwGVFQzqvS/view?usp=sharing) (Blind test set)

mkdir data

put the xsum.zip in data


#### In our experiment, we only use XSum dataset to train and test the t5-small model
Before loading the training set, please pre-tokenize these files  with the following command:
```
mkdir jsonl_files
mkdir tokenized_files
mv data/xsum.zip  ./jsonl_files
cd jsonl_files
unzip xsum.zip && cd ..
python preprocess/preprocess.py --model_name  t5-small --dataset xsum
``` 
This command will produce the tokenized files of XSum `tokenized_files/train.t5.jsonl, tokenized_files/val.t5.jsonl` with the tokenizer of t5-small  

### Training  example for t5-small
#### Important: skip warmup this step, just run the " --warmup Flase" straightly 
Becasuse paper author use 4 A100 to train the model, but we just have only one 3090 with 24G video memoey in our experiment, train the model always raise the error of "cuda out of memory" 
The pretrained model I had downloaded it from https://huggingface.co/t5-small/tree/main, and put it in the `./pretrained_weigths/xsum/t5`
```
#If you have enough graphics cards with sufficient memory and performance, you should use `--warmup True` to train the generation model with NLLLoss 
python run_xsum.py --mode train --gpus 0(,1,2,3) --warmup True --model_name t5-small 
```
with the `./pretrained_weigths/xsum/t5(or pegasus)`, you can skip the warmup
you can set the --validate_every 1 to get the checkpoints quickly
```
#If your don't have enough graphics cards with sufficient memory and performance, please set the -- batch size 8 or 4 to ensure you can run it
python run_xsum.py --mode train --gpus 0(,1,2,3) --warmup False
```

After completing the training process,  several best checkpoints will be stored in a folder named after the training start time by default, for example, `checkpoints/xsum/t5/2023-11-04-15-37-24-196200`

### Generation
We suggest first selecting the best checkpoint based on the dev set with `--mode val` and then generating the results on the test set with the best checkpoint. 

You can run the following command to generate the results on test/dev set with all checkpoints in a given folder, e.g., `checkpoints/xsum/t5/2023-11-04-15-37-24-196200/`:
```
python run_xsum.py --mode val (or test) --model_name t5-small --save_path checkpoints/xsum/t5/2023-11-04-15-37-24-196200/ --gpus 0
```
This will produce the generated results in the floder: `results/xsum/t5/2023-11-04-15-37-24-196200/` containing serval system output and ground truth files: `epoch-0_step-1.val.sys` , `epoch-0_step-1.val.ref`, `epoch-0_step-2.val.sys` , `epoch-0_step-2.val.ref`


To generate the results for test set with  **a specified checkpoint**, you can use the `--ckpt`  parameter and remember to change the mode to `test`:
```
python run_xsum.py --mode test --model_name t5-small --save_path checkpoints/xsum/t5/2023-11-04-15-37-24-196200/ \
--ckpt epoch-2_step-2.pt --gpus 0
```
This will produce the generated results in the floder `results/xsum/t5/2023-11-04-15-37-24-196200/`  containing `epoch-0_step-2.test.sys` , `epoch-0_step-2.test.ref`

### Evaluation
This is an example to evaluate all the generated results for `xsum` in the folder `results/xsum/t5/2023-11-04-15-37-24-196200/`:
```
python evaluation/xsum/eval.py --sys_path results/xsum/t5/2023-11-04-15-37-24-196200/
```
If you only want to evaluate a specified file：
```
python evaluation/xsum/eval.py --sys_file results/xsum/t5/2023-11-04-15-37-24-196200/epoch-0_step-2.sys
```

### Another Example: ToTTo
Because one 3090 can't run the t5-base, always raise the error "cuda out of memory", so I just put the author's steps in here

The first step is downing `totto_meta.zip` via [this link](https://drive.google.com/file/d/1nOlhGKpTWPCmAwmEI_gdALkAXlMn2Tbk/view?usp=sharing) and moving the unziped files to `jsonl_files`. 
```
# preprocess
python preprocess/preprocess.py --model_name  t5-base --dataset totto_meta

# training with warm-up
python run_totto.py --mode train --gpus 0,1,2,3 --warmup True --model_name t5-base  

# generation
python run_totto.py --mode val --model_name t5-base --save_path checkpoints/totto_meta/t5/training_time/ --gpus 0,1,2,3

# evaluation
pip install absl-py (if you do not have the `absl` library)
python evaluation/totto/eval.py --sys_path results/totto_meta/t5/training_time/
```
Results of CoNT-T5-base on *DEV set* of ToTTo are usually `BLEU=48.8~49.2` and `PARENT=58.2~58.7`.  Results of T5-base are `BLEU=47.7` and `PARENT=57.1`. Performance of other scales of T5 can be found in this [paper](https://arxiv.org/pdf/2005.10433.pdf).   
