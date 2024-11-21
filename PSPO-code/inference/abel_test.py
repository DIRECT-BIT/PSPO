import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from typing import Dict,Sequence
import json
import sys
import re
import sklearn
from peft import LoraConfig, PeftModel, get_peft_model

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_QQA":(
        "### question: {question}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "
    ),
    "prompt_AWPNLI": (
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "
        
    ),
    "prompt_NewsNLI": (
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "    
    ),
    "prompt_RTE":(
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n"     
    ),
    "prompt_RedditNLI": (
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n"
    ),
    "prompt_stress":(
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n "
    )
}

# CUDA_VISIBLE_DEVICES=2 python llama_test.py --model_path "/home/ybai/SemEval2024/ppo-weibull/epoch0" --out_folder "/home/ybai/SemEval2024/weibull-test/epoch0/num_beams=3/ --num_beams 3"

import argparse  
parser = argparse.ArgumentParser()
 
parser.add_argument("--model_path", type=str, help="Path to the model")
parser.add_argument("--data_folder", type=str, help="Path to the data folder")   
parser.add_argument("--out_folder", type=str, help="Path to the output folder")   
parser.add_argument("--num_beams", type=int, help="num beams")      

args = parser.parse_args()

model_path = args.model_path
out_folder = args.out_folder
num_beams = args.num_beams
data_folder = args.data_folder

config = LoraConfig(
    r=4, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
# model = get_peft_model(model, config)

tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=512)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"
model.resize_token_embeddings(len(tokenizer))

model.eval()


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s for s in sources]
    examples_tokenized = [_tokenize_fn(strings, tokenizer) for strings in examples]
    
    input_ids = examples_tokenized
    labels = [t for t in targets]
    #import IPython;IPython.embed();exit()
    return dict(input_ids=input_ids, labels=labels)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class MyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(MyDataset, self).__init__()
        print("Loading data...")
        with open(data_path, 'r') as file:
            list_data_dict = json.load(file)

        print("Formatting inputs...")
        self.sources=[]
        
        for example in list_data_dict:

            if example['task'] == 'QP_comment':
                prompt = PROMPT_DICT["prompt_QP_comment"]
            elif example['task'] == 'QP_headline':
                prompt = PROMPT_DICT["prompt_QP_headline"]
            elif example['task'] == 'stressTest':
                prompt = PROMPT_DICT["prompt_stress"]
            elif example['task'] == 'AWPNLI':
                prompt = PROMPT_DICT["prompt_AWPNLI"]
            elif example['task'] == 'NewsNLI':
                prompt = PROMPT_DICT["prompt_NewsNLI"]
            elif example['task'] == 'RTE':
                prompt = PROMPT_DICT["prompt_RTE"]
            elif example['task'] == 'RedditNLI':
                prompt = PROMPT_DICT["prompt_RedditNLI"]
            elif example['task'] == 'QQA':
                prompt = PROMPT_DICT["prompt_QQA"]

            self.sources.append(prompt.format_map(example))


        self.targets = [f"{example['label']}{tokenizer.eos_token}" for example in list_data_dict]

        # print("Tokenizing inputs...")
        # data_dict = preprocess(self.sources, self.targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]
        # import IPython;IPython.embed();exit()

    def __len__(self):
        # return len(self.input_ids)
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        # return self.input_ids[i]
        return self.sources[i]


def extract_ans(response,task):
    match = re.search(r'the answer is option.*(\d+)', response, re.IGNORECASE)
    match2 = re.search(r'the correct answer is \"option (\d+)', response, re.IGNORECASE)
    match3 = re.search(r'the correct answer is option (\d+)', response, re.IGNORECASE)
    match4 = re.search(r'#+\s*(\d+)', response, re.IGNORECASE)
    match5 = re.search(r'\(option (\d+)\)', response, re.IGNORECASE)
    match6 = re.search(r'the correct answer is.*option.*(\d+)', response, re.IGNORECASE)
    match7 = re.search(r'#+\s*([A-Za-z]+)', response, re.IGNORECASE)

    if match:
        result = match.group(1)
  
    elif match2:
        result = match2.group(1)
  
    elif match3:
        result = match3.group(1)
  
    elif match4:
        result = match4.group(1)
  
    elif match5:
        result = match5.group(1)
  
    elif match6:
        result = match6.group(1)
  
    elif match7:
        result = match7.group(1)
        if result == 'A' or result == 'a':
            ans = '1'
        elif result == 'B' or result == 'b':
            ans = '2'
        elif result == 'C' or result == 'c':
            ans = '3'
        else:
            match1 = re.search(r'entailment', response, re.IGNORECASE)
            match2 = re.search(r'contradiction', response, re.IGNORECASE)
            match3 = re.search(r'contradictary', response, re.IGNORECASE)
            match4 = re.search(r'neutral', response, re.IGNORECASE)
            if match1:
                ans = '1'
            elif match2 or match3:
                ans='2'
            elif match4:
                if task == 'NewsNLI' or task == 'RTE':
                    ans='2'
                else:
                    ans='3'
            else:
                ans=result
        try:
            return ans
        except:
            import IPython;IPython.embed();exit()
  
        
    
    else:
        result = "Error"
    try:
        return result
    except:
        import IPython;IPython.embed();exit()


def generate(input_ids):
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,device_map="auto")
    # generated_characters = 0
    # for out in tqdm(pipe(dataset, batch_size=1, return_full_text=False), total=len(dataset)):
    #     print(out[0]["generated_text"])

    # from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
    # from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor

    # logits_processor = LogitsProcessorList()
    # logits_processor.append(MinLengthLogitsProcessor(15, eos_token_id=tokenizer.eos_token))
    # logits_processor.append(InfNanRemoveLogitsProcessor())


    # do_sample=True就会RuntimeError:probability tensor contains either `inf`,`nan` or element<0, next_tokens向量全是nan
    with torch.no_grad():
        generation_outputs = model.generate(
                                input_ids=input_ids,
                                num_beams=num_beams,
                                do_sample=True,
                                min_new_tokens=10,
                                max_new_tokens=512,
                                return_dict_in_generate=True,
                            )
        response=tokenizer.decode(generation_outputs.sequences[0][len(input_ids[0]):],skip_special_tokens=True)
        return response

def run(data_path,out_path,task):
    dataset = MyDataset(data_path,tokenizer)

    total = dataset.__len__()
    with open(out_path, mode='a+', encoding='utf-8') as fout:
        with tqdm(total=total,desc=task,unit='step',position=0) as pbar:
            for i in range(total):
                prompt = dataset[i]
                res = dict()
                res['id'] = str(i)
                inputs = tokenizer(prompt, return_tensors="pt",padding="longest",truncation=True)
                input_ids=inputs["input_ids"].cuda()
                # print(input_ids)
                response = generate(input_ids)
                ans = extract_ans(response,task)

                res['answer'] = ans
                res['generated_text'] = response

                while(res['generated_text'] == ""):
                    print("re-generateing: ",res['id'])
                    res['generated_text'] = generate(input_ids)
                    ans = extract_ans(response,task)

                json.dump(res, fout, ensure_ascii=False)
                fout.write('\n')
                sys.stdout.flush()

                pbar.update(1)




def cal(target_path,pred_path,task):
    targets = []
    with open(target_path,"r+",encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            if task == 'stress' or task == 'QA':
                targets.append(d['label']+1)
            else:
                targets.append(int(d['label'][-1]))

    preds = []
    with open(pred_path,"r+",encoding='utf-8') as f:
        for line in f.readlines():
            try:
                json_object = json.loads(line)
            except:
                print(line)
            if json_object['answer'] == 'Error':
                # id = len(preds)
                # preds.append(targets[id])
                preds.append(-1)
            else:
                try:
                    preds.append(int(json_object['answer'][0]))
                except:
                    preds.append(-1)
                    print(json_object['answer'])

    if task=='headline' or task == 'comment':
        score = sklearn.metrics.f1_score(y_true=targets[:len(preds)], y_pred=preds,average='macro') # QP
    else:
        score = sklearn.metrics.f1_score(y_true=targets, y_pred=preds,average='micro') 
    print(task,score)
    return {task:score}

results=[]

data_path = data_folder + "AWPNLI_test.json"
out_path = out_folder + "AWPNLI.jsonl"
run(data_path,out_path,"AWPNLI")
results.append(cal(data_path,out_path,"AWPNLI"))

data_path = data_folder + "NewsNLI_test.json"
out_path = out_folder + "NewsNLI.jsonl"
run(data_path,out_path,"NewsNLI")
results.append(cal(data_path,out_path,"NewsNLI"))

data_path = data_folder + "Reddit_test.json"
out_path = out_folder + "Reddit.jsonl"
run(data_path,out_path,"Reddit")
results.append(cal(data_path,out_path,"Reddit"))

data_path = data_folder + "RTE_test.json"
out_path = out_folder + "RTE.jsonl"
run(data_path,out_path,"RTE")
results.append(cal(data_path,out_path,"RTE"))

data_path = data_folder + "stress_test.json"
out_path = out_folder + "stress.jsonl"
run(data_path,out_path,"stress")
results.append(cal(data_path,out_path,"stress"))

data_path = data_folder + "QA_test.json"
out_path = out_folder + "QA.jsonl"
run(data_path,out_path,"QA")
results.append(cal(data_path,out_path,"QA"))

print("-----------------------------")
print(out_folder)
print(results)
print("-----------------------------")
with open(out_folder+'results.jsonl','a+') as fw:
    rw = dict()
    rw['dataset'] = model_path
    rw['results'] = results
    json.dump(rw, fw, ensure_ascii=False)
    fw.write('\n')
