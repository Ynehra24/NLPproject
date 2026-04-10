import pandas as pd
df = pd.read_csv('attacked_outputs/agnews_humanizer.csv')
text = df['humanized_text'].iloc[0]
print("Raw repr:")
print(repr(text))
print("Contains actual ZWSP?", '\u200b' in text)
print("Contains literal backslash-u?", '\\u200b' in text)
