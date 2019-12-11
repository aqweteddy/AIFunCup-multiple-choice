import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers.modeling_bert import BertForMultipleChoice, BertConfig
from transformers import BertTokenizer
from utils import InputExample, convert_examples_to_features
import json
import numpy as np
from tqdm import tqdm
import csv


MODEL_NAME = 'bert_funcup_tw'
JSON_FILE = '../data/formal/total_3.json'

# load data
def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


with open(JSON_FILE) as f:
    data = json.load(f)
examples = [InputExample(
    example_id='{}'.format(item['id']),
    question=item['question'],
    contexts=[item['content'], item['content'], item['content'], item['content']],
    endings=[str(item['op1']), str(item['op2']), str(item['op3']), str(item['op4'])],
    label=0
) for item in data]
features = convert_examples_to_features(examples, [0, 1, 2, 3], 512, BertTokenizer.from_pretrained(MODEL_NAME))
all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)

preds = None
out_label_ids = None
model = BertForMultipleChoice.from_pretrained(MODEL_NAME)
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(torch.device('cpu')) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3]}
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

preds = np.argmax(preds, axis=1)
print(preds)
with open(MODEL_NAME+'3.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Answer'])
    for i, pred in enumerate(preds):
       writer.writerow([i+1, pred+1])

# kaggle competitions submit -c 20191130-ai-fun-cup -f submission.csv -m "Message"
