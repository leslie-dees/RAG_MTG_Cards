import sqlite3
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from angle_emb import AnglE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

compute_dtype = torch.float16
cache_path    = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id      = "mobiuslabsgmbh/aanaphi2-v0.1"
model         = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, 
                                                                  cache_dir=cache_path,
                                                                  device_map=device)
tokenizer     = transformers.AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)


vector_db_name = "raw/full_card_vector_database.db"

model.eval();

def semantic_search(query, vector_db_name, number_chunks = 5):
    conn = sqlite3.connect(vector_db_name)
    c = conn.cursor()

    query_embedding = angle.encode(query, to_numpy=True).flatten()

    c.execute("SELECT id, name, card_text, vector FROM vectordb")
    rows = c.fetchall()

    similarities = []
    for row in rows :
        id_, name, card_text, vector_bytes = row
        stored_embedding = np.frombuffer(vector_bytes, dtype=np.float32).flatten()
        sim = 1 - cosine(query_embedding, stored_embedding)
        similarities.append((id_, card_text, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity scores in descending order

    top_matches = similarities[:number_chunks]

    conn.close()

    return [(match[1], match[2]) for match in top_matches]

@torch.no_grad()
def generate(prompt, max_length=1024):
    prompt_chat = prompt
    inputs      = tokenizer(prompt_chat, return_tensors="pt", return_attention_mask=True).to('cuda')
    outputs     = model.generate(**inputs, max_length=max_length, eos_token_id= tokenizer.eos_token_id) 
    text        = tokenizer.batch_decode(outputs[:,:-1])[0]
    return text

def rag_query(query, model, tokenizer):
    chunks = semantic_search(query, vector_db_name, 3)
    
    prompt_prefix = ""
    for chunk in chunks:
        prompt_prefix += chunk[0]+"/n/n"
    
    prompt = f"""### Human: Given the following card data, provide me with the exact text of {query} in the format of :
    \nname: \nmana_cost: \ncmc: \ntype_line: \noracle_text: \npower: \ntoughness: \ncolors: \ncolor_identity: \nkeywords:
    
    \n{prompt_prefix}
    
    Use only the data in the provided chunks above.
   
    ### Assistant:
    """
    
    print(generate(prompt))
    

rag_query("Elesh Norn, Grand Cenobite", None, tokenizer)