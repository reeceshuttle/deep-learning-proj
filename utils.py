import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPTQConfig, AutoModelForCausalLM
# from hf_olmo import OLMoForCausalLM
from functools import partial


from model import OLMoGPTQForCausalLM
import torch.nn as nn

from auto_gptq import BaseQuantizeConfig
# from auto_gptq import AutoGPTQForCausalLM

from constants import step_to_revision


def get_processed_dataset(tokenizer, args):
    def preprocess(example):
        return tokenizer(example['text'], truncation=True, max_length=1024)
    def filter_condition(example):
        return len(tokenizer(example["text"], truncation=True, max_length=1024)["input_ids"]) == 1024
    
    if args.dataset == "c4":
        dataset = load_dataset("c4", "en", split="train", streaming=True)
    else:
        raise NotImplementedError('Implement me!')
    
    filtered_examples = []
    for example in dataset: # getting first .num_samples that have at least 1024 tokens.
        if filter_condition(example):
            filtered_examples.append(example)
        if len(filtered_examples) == args.num_samples:
            break
    dataset = Dataset.from_list(filtered_examples)
    del filtered_examples

    dataset = dataset.map(preprocess, batched=True, remove_columns=['url', 'timestamp', 'text'])
    return dataset

def measure_loss(model, tokenizer, args):
    """
    NOTE: only implemented for batch size 1 right now.
    model: huggingface transformer model
    tokenizer: huggingface tokenizer for model
    args: passed in via command line
        use args are .num_samples, .eval_batch_size, .dataset

    returns 
    avg_loss: float
    """
    def collate_fn(batch):
        text = torch.tensor([item["input_ids"] for item in batch])
        return text
    dataset = get_processed_dataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    model.eval()
    losses = []
    progress_bar = tqdm(range(args.num_samples))
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(args.device) # (batch, seqlen)
            outputs = model(inputs, labels=inputs) # outputs.logits = (batch, seqlen, dictionary)
            # --- use this code if you want a different reduction: ---
            # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            # shift_logits = outputs.logits[..., :-1, :].contiguous()
            # shift_labels = inputs[..., 1:].contiguous()
            # current_loss = loss_fn(shift_logits.permute(0,2,1), shift_labels)
            # --------------------------------------------------------
            current_loss = outputs.loss.item()
            losses.append(current_loss)
            progress_bar.update(1)

    avg_loss = sum(losses)/len(losses)
    return avg_loss


def quantize_model_using_gptq(tokenizer, args):
    gptq_config = GPTQConfig(bits=4, dataset='c4', tokenizer=tokenizer)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, 
        device_map='auto', 
        revision=step_to_revision[args.step], 
        quantization_config=gptq_config
        )
    return quantized_model


def real_quantize_model_using_gptq(tokenizer, args):
    quantize_config = BaseQuantizeConfig(
        bits=4, 
        group_size=128,
        desc_act=True # set to False to speed up interence? but hurts perplexity.
    )
    model = OLMoGPTQForCausalLM.from_pretrained(
        args.base_model_name, 
        revision=step_to_revision[args.step],
        quantize_config=quantize_config
    )

    example = tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
    examples=[{key:example[key] for key in example.keys() if key != 'token_type_ids'}]

    print(f'{examples=}')

    model.quantize(examples)
    import pdb; pdb.set_trace()

    return model


def attach_hooks_for_activation_statistics(model, activations):
    def extract_statistics(outp):
        """
        For a certain sequence, output, the max, min, and percentiles. 
        We will average across these.


        TODO: do these per token and per channel.
        """
        return {
            'max': torch.max(outp).item(),
            'min': torch.min(outp).item(),
            'mean': torch.mean(outp).item(),
                }

    def hook_fn(m, inp, outp, param_name):
        """we will have this hook only operate on the outputs?"""
        result = extract_statistics(outp)
        for key in activations[param_name].keys(): activations[param_name][key].append(result[key])

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f'adding hook for {name}')
            hooks.append(
                module.register_forward_hook(
                    partial(hook_fn, param_name=name)
                )
            )
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


