from utils import *
from sklearn.model_selection import train_test_split
import yaml

with open(Path('data')/'raw'/'config_prepare.yaml', 'r') as file:
    config = yaml.safe_load(file)

download_file(config['pathes']['e3_ligand_db_url'], config['pathes']['e3_ligand_file'])
download_file(config['pathes']['linker_db_url'], config['pathes']['linker_db_file'])
download_file(config['pathes']['warhead_db_url'], config['pathes']['warhead_db_file'])

download_protac_db_metadata(config['pathes']['protac_meta_info_dir'])

protac_db_indice = merge_protac_db_indice(config['pathes']['protac_meta_info_dir'], config['pathes']['protac_db_indice_file'])

protac_df = merge_protac_db_dataset(config['pathes']['protac_editied_csv_file'], config['pathes']['e3_ligand_file'], config['pathes']['linker_db_file'], config['pathes']['warhead_db_file'], config['pathes']['target_protein_half_life_file'], protac_db_indice, out_path=config['pathes']['merged_protac_db_file'])

for idx, row in tqdm(list(protac_df.iterrows())):
    for prefix in ['','e3_ligand_']:
        uniprot_id = row[f'{prefix}uniprot']
        download_protein_file(uniprot_id, config['pathes']['protein_dir'])

# Filter the protac data
protac_df = extract_parameter(protac_df, 'dc50_(nm)', 'min', min_value = 1e-2, max_value= 1e4)
protac_df = extract_parameter(protac_df, 'dmax_(percent)', 'max',min_value= 10, max_value=100)

protac_uniprot_smiles_pair(protac_df, config['pathes']['protac_uniprot_smiles_pair_file'])

protac_df = protac_df.dropna(subset=['dc50_(nm)', 'dmax_(percent)'])

protac_df['e0'] = 1.
protac_df['alpha'] = 1.
protac_df['kcat'] = 1
protac_df['alpha_dmax'] = 1.
protac_df['alpha_dc50'] = 1.

protac_df['kdegp'] = protac_df['k_deg_p']
protac_df['dmax']=protac_df['dmax_(percent)']/100 
protac_df['dc50'] = protac_df['dc50_(nm)']

# get affinity data
affinity_df = pd.read_csv(config['pathes']['affinity_file'])
affinity_df = affinity_df[['protac_id','ㅡlogKd/Ki_target','ㅡlogKd/Ki_e3', 'nM_target', 'nM_e3']]
affinity_df.columns = ['compound_id', 'warhead_binding_energy', 'e3_ligand_binding_energy', 'kdp', 'kde']
protac_df = protac_df.merge(affinity_df, on='compound_id')

protac_df = protac_df.dropna(subset=['dc50', 'dmax', 'kdp', 'kde', 'kdegp'])
argument_cols = list(config['argument'])

target_cols = ['dmax', 'dc50']
cols = config['feature']+argument_cols+target_cols
cols = list(set(cols))
protac_df = protac_df.dropna(subset=cols)
protac_df.to_csv(config['pathes']['protac_file'], index=False)
df_for_datset = protac_df[cols]

# split train and test
train_df, test_df = train_test_split(df_for_datset, test_size=0.2, random_state=42)
train_df.to_csv(config['pathes']['train_file'], index=False)
test_df.to_csv(config['pathes']['test_file'], index=False)

