import pandas as pd
from pathlib import Path

d = Path('attacked_outputs')
for p in d.glob('*_humanizer.csv'):
    df = pd.read_csv(p)
    if 'humanized_text' in df.columns:
        fixed_texts = []
        for val in df['humanized_text']:
            text = str(val)
            if "'humanized_text':" in text and "'best_score':" in text:
                try:
                    # Parse out just the string between the keys
                    part = text.split("'humanized_text': '")[1]
                    txt = part.split("', 'best_score'")[0]
                    
                    # Fix escaped characters that got embedded
                    txt = txt.replace("\\'", "'").replace('\\"', '"')
                    fixed_texts.append(txt)
                except Exception as e:
                    fixed_texts.append(text)
            else:
                fixed_texts.append(text)
        
        df['humanized_text'] = fixed_texts
        df.to_csv(p, index=False)
        print(f"Fixed {p.name}")
