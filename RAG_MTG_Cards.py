import sqlite3
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from angle_emb import AnglE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

vector_db_name = "raw/full_card_vector_database.db"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

def rag_query(query, model, tokenizer):
    chunks = semantic_search(query, vector_db_name, 3)
    
    prompt_prefix = ""
    for chunk in chunks:
        prompt_prefix += chunk[0]+"/n/n"
    
    prompt = f"""[INST]
    Given the following card data, provide me with the exact text of {query} in the format of :
    \nname: \nmana_cost: \ncmc: \ntype_line: \noracle_text: \npower: \ntoughness: \ncolors: \ncolor_identity: \nkeywords:
    
    \n{prompt_prefix}
    
    Use only the data in the provided chunks above.
    """
    
    message = [{
        "role":"user",
        "content": prompt
    }]
    
    model_inputs = tokenizer.apply_chat_template(
        message,
        return_tensors = "pt",
    )
    
    model_inputs = {key: tensor.to(device) for key, tensor in model_inputs.items()}
    
    generated_ids = model.generate(
        model_inputs,
        max_new_tokens = 1000,
        do_sample = True
    )
    
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded)
    
    return decoded[0]

rag_query("Elesh Norn, Grand Cenobite", None, tokenizer)