import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    def preprocess(example):
        return tokenizer(example['text'], truncation=True, max_length=1024)
    def collate_fn(batch):
        text = torch.tensor([item["input_ids"] for item in batch])
        return text
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
