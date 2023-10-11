import os
import requests
import pandas as pd
from transformers import GPT2Tokenizer



df=pd.read_csv('train.csv')
df=df.dropna()
resultdf=pd.DataFrame()
#print(df['Answer'])

input_params=[]
questions=list(df['question'])
answers=list(df['answer'])

for i in range(len(questions)):
    val1=str(questions[i])+'[SEP]'+str(answers[i])
    input_params.append(val1)
    val2=str(answers[i])+'[SEP]'+str(questions[i])
    input_params.append(val2)
    print(f'Completed Question {i}')
resultdf['input_text']=input_params

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
resultdf['tokenized_text'] = resultdf['input_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

max_length=max(resultdf['input_text'].apply(lambda x: len(x)))
resultdf['padded_tokens']=resultdf['tokenized_text'].apply(lambda x: x + [tokenizer.pad_token_id]*(max_length - len(x)))
print(resultdf)
resultdf.to_csv('preprocessedDataBricks.csv',index=False)
