import os
import requests
import pandas as pd
from transformers import GPT2Tokenizer

file="https://docs.google.com/spreadsheets/d/1G9lnEzpdZXortjcGZsh3X6Ul9uRjAIV9kmNk7PYm13s/edit#gid=512763951"
url=file.replace("/edit#gid=","/export?format=csv&gid=")

df=pd.read_csv(url)
df=df.dropna()

#print(df['Answer'])

df['input_text']=df['Question'] + '[SEP]' + df['Answer']
df['input_text']=df['input_text'].apply(lambda x: x.lower().strip())
print(df['input_text'])

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
df['tokenized_text'] = df['input_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

max_length=max(df['input_text'].apply(lambda x: len(x)))
df['padded_tokens']=df['tokenized_text'].apply(lambda x: x + [tokenizer.pad_token_id]*(max_length - len(x)))

df.to_csv('preprocessedData.csv',index=False)
