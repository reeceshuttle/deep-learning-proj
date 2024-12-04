
from hf_olmo import OLMoForCausalLM
from transformers import AutoTokenizer
import torch

from constants import step_to_revision
from utils import measure_loss

def measure_loss_across_training(args):
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    for step in step_to_revision.keys():
        revision_name = step_to_revision[step]
        model = OLMoForCausalLM.from_pretrained(args.base_model_name, revision=revision_name, torch_dtype=torch.float16).to(args.device)
        model.config.pad_token_id = model.config.eos_token_id
        loss = measure_loss(model, tokenizer, args)
        print(f'{step=},{loss=}')
        results[revision_name] = loss
    
    return results
