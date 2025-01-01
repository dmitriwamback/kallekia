import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from PyDictionary import PyDictionary

dictionary = PyDictionary()

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")

# Define a custom logits processor to force specific tokens
class ForceTokenLogitsProcessor:

    def __init__(self, forced_token_ids, weight=5.5):
        self.forced_token_ids = set(forced_token_ids)
        self.weight = weight

    def __call__(self, input_ids, scores):
        for token_id in self.forced_token_ids:
            scores[:, token_id] += self.weight
        return scores


def handler(event, context):

    try:
        data = json.loads(event.get('body', {}))
        prompt = data.get('prompt', '').strip()
        temp = data.get('temp', 1.0)
        nucl = data.get('nucl', 1.0)

        if not prompt:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Prompt is required'})
            }

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Encode the forced word(s)
        words = ['test']
        forced_token_ids = [] 
        for word in words:
            forced_token_ids.extend(tokenizer.encode(word, add_special_tokens=False))

        # Add logits processor
        logits_processor = LogitsProcessorList()
        logits_processor.append(ForceTokenLogitsProcessor(forced_token_ids))

        generated_ids = input_ids[0].tolist()
        max_generated_tokens = 250

        for _ in range(max_generated_tokens):

            outputs = model.generate(
                torch.tensor([generated_ids]),
                max_length=len(generated_ids) + 1,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                do_sample=True,
                top_k=20,
                top_p=nucl,
                temperature=float(temp),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=logits_processor
            )

            new_token = outputs[0, -1].item()
            if new_token == tokenizer.eos_token_id:
                break

            generated_ids.append(new_token)

            new_text = tokenizer.decode([new_token], skip_special_tokens=True)
            meaning = dictionary.meaning(new_text, True)

            print(new_text)
            
            yield f"data: {json.dumps({'data': new_text, 'meaning': 1})}\n\n"

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            'statusCode': 200,
            'body': json.dumps({'generated_text': generated_text})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
