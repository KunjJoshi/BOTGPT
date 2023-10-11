from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import math

print('Loading Model...')
model=GPT2LMHeadModel.from_pretrained('trained_gpt_py')
print('Model Loaded')
print('Loading Tokenizer...')
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
print('Tokenizer Loaded')

test_df=pd.read_csv('Test.csv')
test_df=test_df.dropna()
len_test=len(test_df)
total_perplexity=0

for index,row in test_df.iterrows():
    question=row['QUESTION']

    print(f'\nProcessing Question {index}: {question}')
    answer=row['ANSWER']

    input_ids=tokenizer.encode(question,add_special_tokens=True, return_tensors='pt')

    padding_token_id=tokenizer.eos_token_id
    attention_mask=input_ids.ne(padding_token_id)

    output=model.generate(input_ids, attention_mask=attention_mask, max_length=250, num_return_sequences=1, top_k= 50, top_p=0.95,no_repeat_ngram_size=2)
    decoded_output=tokenizer.decode(output[0], skip_special_tokens=True)

    print(f'\nPredicted Answer: {decoded_output}')

    loss= model(input_ids,labels=input_ids).loss
    perplexity=math.exp(loss.item())
    print(f'\n Calculated Perplexity for Question {index} : {perplexity}')
    total_perplexity=total_perplexity+perplexity

average_perp=total_perplexity/len_test
print(f'\nModel Accuracy: {average_perp}' )



