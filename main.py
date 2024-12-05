from hf_olmo import OLMoForCausalLM
import argparse
from transformers import AutoTokenizer
import torch
import torch.nn as nn

from constants import step_to_revision
from experiments import measure_loss_across_training
from utils import *


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--step", type=str, required=True)
    argsparser.add_argument("--dataset", type=str, required=False, default="c4")
    argsparser.add_argument("--num_samples", type=int, required=False, default=100)
    argsparser.add_argument("--eval_batch_size", type=int, required=False, default=8)
    args = argsparser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "allenai/OLMo-1B"
    args.device = device
    args.base_model_name = base_model_name

    assert args.eval_batch_size == 1, "higher batch sizes arent supported yet"

    revision_name = step_to_revision[args.step]
    model = OLMoForCausalLM.from_pretrained(base_model_name, revision=revision_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    # measure_loss(model, tokenizer, args)

    # model = real_quantize_model_using_gptq(tokenizer, args)
    # model.config.pad_token_id = model.config.eos_token_id

    # measure_loss(model, tokenizer, args)

    activations = {
            name: {
            'max':[],
            'min':[],
            'mean':[],
            } for name, module in model.named_modules() if isinstance(module, nn.Linear)
            }
        
    hooks = attach_hooks_for_activation_statistics(model, activations)
    
    loss = measure_loss(model, tokenizer, args)
    print(f'average loss: {loss}')
    
    remove_hooks(hooks)

    def postprocess_activations():
        raise NotImplementedError('Implement me!')

    import pdb; pdb.set_trace()
