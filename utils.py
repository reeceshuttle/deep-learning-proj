import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
# from hf_olmo import OLMoForCausalLM
from functools import partial

import torch.nn as nn

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

def measure_perplexity(model, tokenizer, args):
    return torch.exp(measure_loss(model, tokenizer, args))

def evaluate_ppl(model, tokenizer):
    return torch.exp(evaluate(model, tokenizer))

def evaluate_loss(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()

    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.stack(nlls).sum() / (nsamples * 2048)


def attach_hooks_for_activation_statistics(model, activations):
    def extract_statistics(outp):
        """
        For a certain sequence, output, the max, min, and percentiles. 
        We will average across these.


        TODO: do these per token and per channel.
        """
        # print(outp.shape)

        # shape: (batch, seqlen, hidden)

        # code to see the first sentences vals for each channel
        # sorted(activations['model.transformer.blocks.18.ff_out']['max'][0])

        return {
            'max': torch.max(outp, dim=1).values.tolist()[0],
            'min': torch.min(outp, dim=1).values.tolist()[0],
            'mean': torch.mean(outp, dim=1).tolist()[0],
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

def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    """From TinyML Pset 4."""
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        # if n_samples < 10: print(f'{samples=},{n_run=}')
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    # print(f'{cat_samples.shape[1]=}')
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

@torch.no_grad()
def get_calib_feat(model, tokenizer):
    """From TinyML Pset 4."""
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return input_dict

def eval_using_harness(model, eval_tasks):
    import lm_eval

    # avoiding annoying print statement:
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    result = {}

    lm_for_eval = lm_eval.models.huggingface.HFLM(
        pretrained=model,                  
        batch_size=32) 
    
    results = lm_eval.evaluator.simple_evaluate(
        lm_for_eval,
        tasks=eval_tasks,
        num_fewshot=0,
        verbosity="CRITICAL",
    )
    for task in eval_tasks:
        print(f"{task}: acc: {results['results'][task]['acc,none']}")
        result[f"{task},acc"] = results['results'][task]['acc,none']

    return result

def get_model_activations(model, tokenizer):
    """
    Gets the models activations for one example.
    """
    activations = {}

    def get_activations(m, x, y, name):
        # print(f'in get_activations, {name}')
        if isinstance(x, tuple):
            x = x[0]
        assert name+',in' not in activations, "show be only doing 1 sequence"
        activations[name+',in'] = x.clone()
        activations[name+',out'] = y.clone()
    
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # print(f'registered for {name}')
            hooks.append(
                m.register_forward_hook(
                    partial(get_activations, name=name)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer, n_samples=64)
    # print(f'samples1:{samples}')
    samples = [samples[0]]
    # print(f'samples2:{samples}')
    pbar = tqdm(samples)
    for input_ids in pbar:
        # print(f'in loop')
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return activations
    

