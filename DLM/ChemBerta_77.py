import pandas as pd
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_embeddings(smiles_list, model, tokenizer, device, batch_size=32):
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(smiles_list), batch_size)):
        batch_smiles = smiles_list[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Use mean pooling of the last hidden state as the embedding
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Masked mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = sum_embeddings / sum_mask
        
        embeddings.append(batch_embeddings.cpu().numpy())
        
    return np.vstack(embeddings)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # 1. Process DAVIS unique drugs
    # davis_csv = '/HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/dataset/unique_drugs_davis.csv'
    # if os.path.exists(davis_csv):
    #     print("\nExtracting embeddings for DAVIS...")
    #     df_davis = pd.read_csv(davis_csv)
    #     smiles = df_davis['compound_iso_smiles'].tolist()
    #     drug_names = df_davis['drug_name'].tolist()
    #     embeddings = get_embeddings(smiles, model, tokenizer, device)
        
    #     save_path = '/HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/Pretrained_embedding/ChemBERTa_77/davis_embeddings_chemberta.npz'
    #     # Requested keys: SMILES, drug_name, drug_embedding
    #     np.savez(save_path, SMILES=smiles, drug_name=drug_names, drug_embedding=embeddings)
    #     print(f"Saved DAVIS embeddings to {save_path}, shape: {embeddings.shape}")
    
    # 2. Process ChEMBL unique drugs
    chembl_csv = '/HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/Mutant_test/unique_drugs.csv'
    if os.path.exists(chembl_csv):
        print("\nExtracting embeddings for ChEMBL...")
        df_chembl = pd.read_csv(chembl_csv)
        smiles_chembl = df_chembl['canonical_smiles'].tolist()
        # ChEMBL usually uses ID as the name
        id_col = 'compound_chembl_id' if 'compound_chembl_id' in df_chembl.columns else 'canonical_smiles'
        ids_chembl = df_chembl[id_col].tolist()
        embeddings_chembl = get_embeddings(smiles_chembl, model, tokenizer, device)
        
        save_path = '/HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/Pretrained_embedding/ChemBERTa_77/Mut_chembl_embeddings_chemberta.npz'
        # Requested keys: SMILES, drug_name, drug_embedding
        np.savez(save_path, SMILES=smiles_chembl, drug_name=ids_chembl, drug_embedding=embeddings_chembl)
        print(f"Saved ChEMBL embeddings to {save_path}, shape: {embeddings_chembl.shape}")

if __name__ == "__main__":
    main()



# python chembert_embedding_concat.py  
# --emb_path /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/Kd_embeddings.npz   
# --drug_emb_path /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/Pretrained_embedding/ChemBERTa_77/chembl_embeddings_chemberta.npz   
# --davis_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/dataset/chembl_35/filtered_Kd/all_data   
# --save_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_100/Kd &&

# python chembert_embedding_concat.py   
# --emb_path /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/Ki_embeddings.npz   
# --drug_emb_path /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/Pretrained_embedding/ChemBERTa_77/chembl_embeddings_chemberta.npz   
# --davis_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/dataset/chembl_35/filtered_Ki/all_data   
# --save_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_100/Ki && 

# python chembert_embedding_concat.py   
# --emb_path /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/IC50_embeddings.npz   
# --drug_emb_path /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/Pretrained_embedding/ChemBERTa_77/chembl_embeddings_chemberta.npz   
# --davis_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/dataset/chembl_35/filtered_IC50/all_data   
# --save_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_100/IC50