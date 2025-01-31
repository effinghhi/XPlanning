import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens
import os


# determine relative path of model folder
model_path_rel = os.path.relpath("/mnt/models/Llama-2-7b-hf", os.getcwd())

model = LlamaForCausalLM.from_pretrained(model_path_rel, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation='eager', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_path_rel, local_files_only=True )

# apply AttnLRP rules
attnlrp.register(model)

prompt ="Sarah went to the store to buy some milk. After picking up a gallon, she noticed a sale on apples. Deciding to take advantage of it, she bought four apples. As she walked home, she counted her bags and realized she was carrying two. How many apples did Sarah buy? The answer is: "

input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

n_iter = 2 #30
total_relevance = 0
for i in range(n_iter):
    input_embeds = model.get_input_embeddings()(input_ids)
    
    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)
    
    max_logits.backward(max_logits)
    relevance = input_embeds.grad.float().sum(-1).cpu()[0]
    


    # normalize relevance between [-1, 1] for plotting
    relevance = relevance / relevance.abs().max()
    
    if (i == 0):
        total_relevance = relevance
    else:
        total_relevance = torch.cat((total_relevance, torch.tensor([0])), dim=-1)
        total_relevance += relevance


    
    max_prediction_token= tokenizer.decode([max_indices])
    print("the model says: ", max_prediction_token)
    print("max index: ", max_indices.to("cpu").item() , " and eos token index: ", tokenizer.eos_token_id)
    
    if (max_indices.to("cpu").item() == tokenizer.eos_token_id or 29889 == max_indices.to("cpu").item()):
        break
    
    if (i < n_iter-1):
        input_ids =  torch.cat((input_ids, torch.tensor([[max_indices]]).to(model.device)), dim=1)

# remove '_' characters from token strings
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)



# normalize relevance between [-1, 1] for plotting
total_relevance = total_relevance /total_relevance.abs().max()
pdf_heatmap(tokens, total_relevance, path='/mnt/output/heatmap.pdf', backend='xelatex')
