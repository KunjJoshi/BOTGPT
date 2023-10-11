from transformers import GPT2LMHeadModel, GPT2Tokenizer

model=GPT2LMHeadModel.from_pretrained('trained_gpt_py')
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')

user_question=input()

print('Processing Question : ',user_question)

input_ids=tokenizer.encode(user_question,add_special_tokens=True, return_tensors='pt')

print('Input_IDS ',len(input_ids))

padding_token_id=tokenizer.eos_token_id
attention_mask=input_ids.ne(padding_token_id)

output=model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

print('Received Output: ', len(output))
response=tokenizer.decode(output[0], skip_special_tokens=True)

print(response)