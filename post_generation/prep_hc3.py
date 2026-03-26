import json, urllib.request, random, os
print('Downloading dataset...')
url = 'https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl'
urllib.request.urlretrieve(url, 'post_generation/hc3_all.jsonl')
print('Processing...')
ai_texts, human_texts = [], []
with open('post_generation/hc3_all.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row.get('chatgpt_answers'):
            for ans in row['chatgpt_answers']:
                if isinstance(ans, str) and len(ans) > 50:
                    ai_texts.append(ans.replace('\n', ' ').strip())
        if row.get('human_answers'):
            for ans in row['human_answers']:
                if isinstance(ans, str) and len(ans) > 50:
                    human_texts.append(ans.replace('\n', ' ').strip())
random.seed(42)
random.shuffle(ai_texts)
random.shuffle(human_texts)
os.makedirs('post_generation/data', exist_ok=True)
train_ai = ai_texts[:int(len(ai_texts)*0.8)]
eval_ai = ai_texts[int(len(ai_texts)*0.8):]
with open('post_generation/data/train_ai_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_ai) + '\n')
with open('post_generation/data/eval_ai_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(eval_ai) + '\n')
with open('post_generation/data/human_corpus.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(human_texts) + '\n')
print(f'Done. Train AI: {len(train_ai)}, Eval AI: {len(eval_ai)}, Human: {len(human_texts)}')
