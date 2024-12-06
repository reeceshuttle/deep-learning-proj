from hf_olmo import OLMoForCausalLM
import argparse
from transformers import AutoTokenizer
import torch
import torch.nn as nn

from constants import step_to_revision
from experiments import measure_loss_across_training
from quantization import pseudo_quantize_model
from utils import *


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--size", type=str, required=True)
    argsparser.add_argument("--step", type=str, required=True)
    argsparser.add_argument("--dataset", type=str, required=False, default="c4")
    argsparser.add_argument("--num_samples", type=int, required=False, default=100)
    argsparser.add_argument("--eval_batch_size", type=int, required=False, default=8)
    args = argsparser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.size == '1b':
        base_model_name = "allenai/OLMo-1B"
    elif args.size == '7b':
        base_model_name = "allenai/OLMo-7B"
    else:
        raise ValueError
    args.device = device
    args.base_model_name = base_model_name

    assert args.eval_batch_size == 1, "higher batch sizes arent supported yet"

    revision_name = step_to_revision[args.base_model_name][args.step]
    # model = OLMoForCausalLM.from_pretrained(base_model_name, revision=revision_name, torch_dtype=torch.float16).to(device)
    args.base_model_name = 'meta-llama/Meta-Llama-3-8B'
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    # measure_loss(model, tokenizer, args)

    # model = real_quantize_model_using_gptq(tokenizer, args)
    # model.config.pad_token_id = model.config.eos_token_id

    # measure_loss(model, tokenizer, args)
    do_activations = False
    if do_activations:
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
    
    loss = measure_loss(model, tokenizer, args)
    print(f'pre quantization (16 bit) loss: {loss}')

    han_ppl = evaluate(model, tokenizer)
    print(f'han_ppl:{han_ppl}')

    # to show that quantization error grows across amount of quantization:
    for quantization_bits in [8, 7, 6, 5, 4, 3, 2]:
        # model = OLMoForCausalLM.from_pretrained(base_model_name, revision=revision_name, torch_dtype=torch.float16).to(device)
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.float16).to(device)

        # loss = measure_loss(model, tokenizer, args)
        # print(f'pre quantization (16 bit) loss: {loss}')
        pseudo_quantize_model(model, w_bit=quantization_bits, q_group_size=128)
        loss = measure_loss(model, tokenizer, args)
        print(f'post quantization ({quantization_bits} bit) loss: {loss}')
        han_ppl = evaluate(model, tokenizer)
        print(f'han_ppl:{han_ppl}')
    

    import pdb; pdb.set_trace()
