from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds_sft = load_dataset("Chtistina777/GAP-MHQA-SFT-7K")
ds_sft.save_to_disk("./GAP-MHQA-SFT-Dataset")

ds_rl = load_dataset("Chtistina777/GAP-MHQA-RL")
ds_rl.save_to_disk("./GAP-MHQA-RL-Dataset")