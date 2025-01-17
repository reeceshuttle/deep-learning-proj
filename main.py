from hf_olmo import OLMoForCausalLM
import argparse
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import os
import json

from constants import step_to_revision
from experiments import measure_loss_across_training
from quantization import pseudo_quantize_model_naive, pseudo_quantize_model_awq
from utils import *

def load_model(args):
    model = OLMoForCausalLM.from_pretrained(args.base_model_name, 
                                            revision=args.revision_name, 
                                            torch_dtype=torch.float32).to(args.device)
    model.config.pad_token_id = model.config.eos_token_id
    return model

if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--size", type=str, required=True)
    argsparser.add_argument("--step", type=str, required=True)
    argsparser.add_argument("--dataset", type=str, required=False, default="c4")
    argsparser.add_argument("--num_samples", type=int, required=False, default=100)
    argsparser.add_argument("--eval_batch_size", type=int, required=False, default=1)
    argsparser.add_argument("--study_activations", type=bool, required=False, default=False)
    argsparser.add_argument("--measure_task_performance", type=bool, required=False, default=False)
    argsparser.add_argument("--quantization_method", type=str, required=False, default=False, help="pick one of \'awq\',\'naive\'")
    # argsparser.add_argument("--do_eval_harness", type=bool, required=False, default=False)
    argsparser.add_argument("--reproduce_paper", type=bool, required=False, default=False)
    argsparser.add_argument("--measure_quantization_error", type=bool, required=False, default=False)
    argsparser.add_argument("--measure_activation_statistics", type=bool, required=False, default=False)
    argsparser.add_argument("--dont_measure_losses", action='store_true')
    argsparser.add_argument("--dont_measure_tasks", action='store_true')
    argsparser.add_argument("--study_sparsity", action='store_true')
    args = argsparser.parse_args()
    print(args)

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


    args.revision_name = step_to_revision[args.base_model_name][args.step]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.study_activations:
        model = load_model(args)
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
    
    if args.measure_activation_statistics:
        stats = {}
        statistics_dir_path = os.path.join(os.path.dirname(__file__), 'statistics')
        os.makedirs(statistics_dir_path, exist_ok=True)
        model = load_model(args)
        activations = get_model_activations(model, tokenizer, n_samples=64)
        # for the fp16 model, do the forward passes and measure the stats per matrix: min, max, mean, #(>n),
        for key in activations.keys():
            stats[key] = {
                'max': torch.max(activations[key]).item(),
                'min': torch.min(activations[key]).item(),
                'abs_max': torch.max(torch.abs(activations[key])).item(),
                'mean': torch.mean(activations[key]).item(),
                'abs_mean': torch.mean(torch.abs(activations[key])).item(),
                '>1': torch.sum(1*(activations[key]>1)).item(),
                '>5': torch.sum(1*(activations[key]>5)).item(),
                '>10': torch.sum(1*(activations[key]>10)).item(),
                '>50': torch.sum(1*(activations[key]>50)).item(),
                '>100': torch.sum(1*(activations[key]>100)).item(),
                '>500': torch.sum(1*(activations[key]>500)).item(),
                '>1000': torch.sum(1*(activations[key]>1000)).item(),
            }
        
        print(stats)
        with open(os.path.join(statistics_dir_path, f'{args.size},{args.step}.json'), "w") as f:
            json.dump(stats, f)
        
    
    if args.measure_quantization_error:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        activations_dir_path = os.path.join(os.path.dirname(__file__), 'errors')
        os.makedirs(activations_dir_path, exist_ok=True)
        all_activations = {}
        results = {}
        bits_to_use = [16, 5, 4, 3]
        for w_bits in bits_to_use:
            print(f'starting for {w_bits=}')
            model = load_model(args)
            print('quantizing...')
            if args.quantization_method == 'naive':
                    print('doing naive...')
                    pseudo_quantize_model_naive(
                        model,
                        w_bit=w_bits,
                        q_group_size=128
                    )
            elif args.quantization_method == 'awq':
                print('doing awq...')
                input_feat = get_calib_feat(model, tokenizer)
                pseudo_quantize_model_awq(
                    model,
                    w_bit=w_bits,
                    input_feat=input_feat,
                    a_bit=16,
                    q_group_size=128
                )
            else:
                raise ValueError
            print(f'getting model activations for {w_bits=}...')
            model = model.to(torch.float16) # trying to avoid OOM error
            modify_inputs_of_model(model)
            # do_in_activations = False
            all_activations[w_bits] = get_model_activations(model, tokenizer, n_samples=64)
            del model

            if w_bits in bits_to_use[1:]:
                results[w_bits] = {}
                for key in all_activations[w_bits].keys():
                    norm = torch.norm(torch.abs(all_activations[w_bits][key]-all_activations[16][key]))
                    max_diff = torch.max(torch.abs(all_activations[w_bits][key]-all_activations[16][key]))
                
                    results[w_bits][key] = {
                        'error_norm': norm.item(),
                        'error_max_diff':max_diff.item()
                    }
                del all_activations[w_bits]
            print(f'{all_activations.keys()=}, {results.keys()=}')
        
        with open(os.path.join(activations_dir_path, f'{args.size},{args.step},{args.quantization_method}.json'), "w") as f:
            json.dump(results, f)
        

        # will use one example and store the full everything for each
    
    # two things, one will measure stats of activations across models. do this across 2048 examples.


    # then measure quantization error between fp16 and quantized activaitons (before or after every param?).
    # (will need to store the fp16 activations for each forward pass). how many examples to do? just one? 10 examples?
    
    if args.reproduce_paper:
        
        result_dir_path = os.path.join(os.path.dirname(__file__), 'results')
        result_name = os.path.join(result_dir_path, f'{args.size},{args.step},{args.quantization_method}.json')
        if not os.path.exists(result_dir_path):
            os.makedirs(result_dir_path, exist_ok=True)
        
        if not os.path.exists(result_name):
            results = {}
            print('new experiment type...')
        else:
            with open(result_name, 'r') as file:
                results = json.load(file)
            print('loaded existing results..')
        
        for w_bits in [16, 5, 4, 3, 2]:
            print(f'doing {w_bits} bit for {args.size},{args.step}')
            model = load_model(args)
            if args.quantization_method == 'naive':
                print('doing naive...')
                pseudo_quantize_model_naive(
                    model,
                    w_bit=w_bits,
                    q_group_size=128
                )
            elif args.quantization_method == 'awq':
                print('doing awq...')
                input_feat = get_calib_feat(model, tokenizer)
                pseudo_quantize_model_awq(
                    model,
                    w_bit=w_bits,
                    input_feat=input_feat,
                    a_bit=16,
                    q_group_size=128
                )
            else:
                raise ValueError
            
            to_add = {}
            if not args.dont_measure_losses:
                print('measuring losses...')
                my_loss = measure_loss(model, tokenizer, args)
                han_loss = evaluate_loss(model, tokenizer).item()
                to_add = {
                    'my_loss': my_loss,
                    'han_loss': han_loss,
                }
            if not args.dont_measure_tasks:
                print('measuring tasks...')
                eval_tasks = ['hellaswag'] # 'piqa', 'winogrande', 'lambada_openai', 
                accs = eval_using_harness(model, eval_tasks)
                to_add = {**to_add, **accs}
            
            w_bits = str(w_bits) # json saves all keys as strs
            if w_bits not in results:
                print('did not find key...')
                results[w_bits] = {
                    **to_add
                }
            else:
                print(f'found key...')
                results[w_bits] = {
                    **results[w_bits], **to_add
                }

            print(results)
        print(args.base_model_name, args.step)
        print(results)
        with open(result_name, "w") as f:
            json.dump(results, f)
    
    if args.study_sparsity:
        sparse_dir_path = os.path.join(os.path.dirname(__file__), 'sparsity')
        sparse_name = os.path.join(sparse_dir_path, f'{args.size},{args.step}.json')
        if not os.path.exists(sparse_dir_path):
            os.makedirs(sparse_dir_path, exist_ok=True)

        results = {}
        model = load_model(args)

        magnitudes = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1]
        threshold_result = {magnitude:0 for magnitude in magnitudes}

        quantiles = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]
        quantile_results = {}

        for n,m in model.named_modules():
            if isinstance(m, nn.Linear):
                abs_w = torch.abs(m.weight)
                for magnitude in magnitudes:
                    threshold_result[magnitude] += torch.sum(torch.abs(m.weight) < magnitude).item()
                
                quantile_results[n] = {}
                num_to_sample = 1000000
                flat_weights = abs_w.flatten()
                sampled_indices = torch.randperm(flat_weights.size(0))[:num_to_sample]
                sampled_weights = flat_weights[sampled_indices]
                for quantile in quantiles:
                    quantile_results[n][quantile] = torch.quantile(sampled_weights, quantile).item()
        
        final_results = {'thresholds': threshold_result, 'approx_quantiles': quantile_results}
        print(final_results)
        with open(sparse_name, "w") as f:
            json.dump(final_results, f)
