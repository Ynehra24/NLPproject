import pandas as pd
from humanizer import humanize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

print("\n==========================================")
print(" ⚡ FAST EVAL: 10 SAMPLES WHITE-BOX CSBP ⚡")
print("==========================================\n")

print("Loading BERT-Base & Tokenizer...")
model_id = "textattack/bert-base-uncased-ag-news"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).to("mps")
model.eval()

# Load 10 unattacked texts from the existing humanizer CSV (original_text column)
print("Loading 10 samples from agnews_humanizer.csv...")
df = pd.read_csv("/Users/yatharthnehva/NLPproject/characterlevelattacks/coreattacks/attacked_outputs/agnews_humanizer_bert.csv")

successes = 0
total = 0

print("\nStarting Humanizer Attacks...")
for i, row in df.iterrows():
    orig_col = 'original_text' if 'original_text' in df.columns else 'text'
    text = str(row[orig_col])
    
    # 1. Check original label probability
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to("mps")
        logits = model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]
        orig_class = probs.argmax().item()
        orig_prob  = probs[orig_class].item()
    
    print(f"\n[Sample {i+1}] Original | Class: {orig_class} ({orig_prob*100:.1f}%)")
    
    # 2. Attack WITH the classifier wired in!
    # By passing target_model, CSBP automatically runs pipeline() and directly targets the exact words BERT relies on.
    result = humanize(
        text, 
        iterations=5, 
        beam_width=3, 
        n_candidates=10, 
        target_model=model_id, # THIS IS THE SECRET SAUCE!
        device_override="mps"
    )
    
    attacked = result.get('humanized_text', result.get('best_text', text))
    
    # 3. Check new label probability
    with torch.no_grad():
        inputs2 = tokenizer(attacked, return_tensors='pt', truncation=True, max_length=512).to("mps")
        logits2 = model(**inputs2).logits
        probs2 = logits2.softmax(dim=-1)[0]
        new_class = probs2.argmax().item()
        new_prob  = probs2[orig_class].item()
        
    print(f"[Sample {i+1}] Attacked | New Class: {new_class} (Original Class dropped to {new_prob*100:.1f}%)")
    
    total += 1
    if orig_class != new_class:
        print("  -> ✅ FLIPPED!")
        successes += 1
    else:
        print("  -> ⚠️ FAILED TO FLIP.")

print("\n==========================================")
print(f" FINAL ASR ON 10 SAMPLES: {successes}/{total} ({successes/total*100:.1f}%)")
print("==========================================")
