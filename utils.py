from collections import ChainMap
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import requests
import re
from typing import Literal
import json
import requests
from pathlib import Path
from tqdm import tqdm

class Prefix(Enum):
    LINKER = 'linker_'
    WARHEAD = 'warhead_'
    E3_LIGAND = 'e3_ligand_'

def download_protac_db_metadata(protac_meta_info_dir, max_id = 6120):
    if not isinstance(protac_meta_info_dir, Path):
        protac_meta_info_dir = Path(protac_meta_info_dir)
    protac_meta_info_dir.mkdir(parents=True,exist_ok=True)
    
    for i in tqdm(range(max_id), desc='Downloading ProtacDB metadata'):
        try :
            save_path = protac_meta_info_dir/ f'{i}.json'
            if save_path.exists(): continue

            session = requests.Session()
            url = f"http://cadd.zju.edu.cn/protacdb/compound/dataset=protac&id={i}"
            response = session.get(url)

            xsrf_token = session.cookies.get('_xsrf')
            post_data = {
            '_xsrf': xsrf_token
        }
            headers = post_data

            post_url = f"http://cadd.zju.edu.cn/protacdb/compound/dataset=protac&id={i}"
            post_response = session.post(post_url, data=post_data, headers=headers)
            if post_response.ok :
                with open(save_path, 'w') as file:
                    file.write(post_response.text)
        except :
            with open(save_path, 'w') as file: 
                file.write('')

def merge_protac_db_indice(protac_meta_info_dir, protac_db_indice_path: Path):
    if not isinstance(protac_meta_info_dir, Path):
        protac_meta_info_dir = Path(protac_meta_info_dir)
    if not isinstance(protac_db_indice_path, Path):
        protac_db_indice_path = Path(protac_db_indice_path)

    if protac_db_indice_path.exists():
        return pd.read_csv(protac_db_indice_path)
    protac_db_indice = pd.DataFrame(columns=['protac_id','warhead_id','e3_ligand_id'])
    for p in tqdm(list(protac_meta_info_dir.iterdir()), desc='Merging ProtacDB indices'):
        with open(p, 'r') as file :
            try :
                data = file.read()
                if data == '':
                    pass
                else :
                    data = json.loads(data)
                    if type(data)==dict and data.get('content', '') == "no": continue
                    data = dict(ChainMap(*data))
                    linker_id = '0' if data['id_linker'] is None else data['id_linker']
                    row = pd.DataFrame({
                'protac_id': [p.stem],
                'warhead_id':[data['id_warhead']],
                'linker_id': [linker_id],
                'e3_ligand_id':[data['id_e3_ligand']]}).astype(int)
                    protac_db_indice = pd.concat([protac_db_indice, row], ignore_index=True)
            except Exception as e:
                print(e, p)

    protac_db_indice = protac_db_indice.sort_values('protac_id', ignore_index=True).astype(int)
    protac_db_indice.to_csv(protac_db_indice_path, index=False)
    return protac_db_indice

def merge_protac_db_dataset(protac_csv_file, e3_ligand_file, linker_db_file, warhead_db_file, target_protein_half_life_file, protac_db_indice : pd.DataFrame, out_path: Path):
    if not isinstance(protac_csv_file, Path):
        protac_csv_file = Path(protac_csv_file)
    if not isinstance(e3_ligand_file, Path):
        e3_ligand_file = Path(e3_ligand_file)
    if not isinstance(linker_db_file, Path):
        linker_db_file = Path(linker_db_file)
    if not isinstance(warhead_db_file, Path):
        warhead_db_file = Path(warhead_db_file)
    if not isinstance(target_protein_half_life_file, Path):
        target_protein_half_life_file = Path(target_protein_half_life_file)
    if not isinstance(out_path, Path):
        out_path = Path(out_path)

    if out_path.exists():
        return pd.read_csv(out_path)
    protac_df = read_csv_and_clean_column_name(protac_csv_file)
    e3_ligand_df = read_csv_and_clean_column_name(e3_ligand_file)
    linker_df = read_csv_and_clean_column_name(linker_db_file)
    warhead_df = read_csv_and_clean_column_name(warhead_db_file)

    linker_df = linker_df.rename(columns={'compound_id': 'id'})
    linker_df['id'] = linker_df['id'].astype(int)
    linker_df = linker_df.add_prefix(Prefix.LINKER.value)

    warhead_df = warhead_df.rename(columns={'compound_id': 'id'})
    warhead_df['id'] = warhead_df['id'].astype(int)
    warhead_df = warhead_df.add_prefix(Prefix.WARHEAD.value)

    e3_ligand_df = e3_ligand_df.rename(columns={'compound_id': 'id'})
    e3_ligand_df['id'] = e3_ligand_df['id'].astype(int)
    e3_ligand_df = e3_ligand_df.add_prefix(Prefix.E3_LIGAND.value)

    half_life_df = pd.read_csv(target_protein_half_life_file)

    protac_db_indice = protac_db_indice.rename(columns={'protac_id':'compound_id'})
    protac_df = protac_df.merge(protac_db_indice, on='compound_id')
    protac_df = protac_df.merge(linker_df, on='linker_id')

    warhead_df_for_merge = warhead_df.drop_duplicates(subset=['warhead_id'])
    protac_df = protac_df.merge(warhead_df_for_merge, on='warhead_id')


    e3_ligand_df['e3_merge_id'] = e3_ligand_df['e3_ligand_id'].astype(str) + e3_ligand_df['e3_ligand_target']
    protac_df['e3_merge_id'] = protac_df['e3_ligand_id'].astype(str) + protac_df['e3_ligase']

    protac_df = protac_df.merge(e3_ligand_df, on='e3_merge_id')
    half_life_mean = half_life_df['half_life'].mean()
    half_life_df['half_life'] = half_life_df['half_life'].fillna(half_life_mean)
    half_life_df['k_deg_p'] = 0.693/half_life_df['half_life']

    protac_df = protac_df.merge(half_life_df, left_on='uniprot', right_on='uniprot', how='left')
    return protac_df

def read_csv_and_clean_column_name(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.replace(', ', ',').str.lower()
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    df.columns = df.columns.str.replace('%', 'percent').str.lower()
    df['compound_id'] = df['compound_id'].astype(int)
    return df

def download_file(url: str, save_path: Path, forced : bool = False):
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not forced:
        return
    print(f"Downloading: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with save_path.open("wb") as file:
            file.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {url}")

def extract_parameter(
        df : pd.DataFrame, 
        column_name : str, 
        how: Literal['max', 'min', 'mean'], 
        min_value: float, 
        max_value: float,
        na_list: set[str] = {'N.D.','nan'}) -> pd.DataFrame:
    df[column_name] = df[column_name].astype(str)
    
    for na in na_list:
        df = df.loc[~df[column_name].str.contains(na, na=False)]
    if len(na_list) > 0:
        df = df.dropna(subset=[column_name])
    def extract_with_re(values:str):
        numbers = re.findall(r'\d+\.?\d*', values)
        numbers = map(float, numbers)
        numbers = [x for x in numbers if x > min_value]
        numbers = [x for x in numbers if x < max_value]
        if numbers: 
            if how == 'min':
                return min(numbers)
            elif how == 'max':
                return max(numbers)
            elif how == 'mean':
                return np.mean(numbers)
        else: 
            return None

    df[column_name] = df[column_name].apply(extract_with_re)
    return df


def protac_uniprot_smiles_pair(protac_df_filtered, protac_uniprot_smiles_pair_file):
    result_df = protac_df_filtered[['compound_id','uniprot','e3_ligand_uniprot','smiles', 'e3_ligand_id_x']]

    column_update_dict = {
    'compound_id':'protac_id',
    'uniprot':'target_uniprot',
    'smiles':'protac_smiles',
    'e3_ligand_id_x':'e3_ligand_id'
}
    result_df = result_df.rename(columns=column_update_dict)
    result_df = result_df.drop_duplicates()
    result_df = result_df.dropna()
    print(len(result_df))
    result_df.to_csv(protac_uniprot_smiles_pair_file, index=False)

def download_protein_file(uniprot_id : str, dir_path : Path, forced = False):
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    dir_path.mkdir(parents=True, exist_ok=True)

    url = urlparse(f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb')
    file_name = url.path.split("/")[-1]
    file_path = dir_path / file_name

    if file_path.exists() and not forced:
        return
    
    if str(uniprot_id) == 'nan' or uniprot_id is None :
        return

    response = requests.get(url.geturl())
    if response.status_code == 200:
        with file_path.open("wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download: {url}")