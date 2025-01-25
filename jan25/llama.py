
from torch.nn.parallel import DistributedDataParallel as DDP

def initialize_distributed_training(model_parallel_size: Optional[int] = None):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Return local rank and world size for DDP
    return local_rank

def build(ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int, model_parallel_size: Optional[int] = None, seed: int = 1) -> "Llama":
    """
    Build a Llama instance and load the model checkpoint.
    """
    # Initialize distributed training
    local_rank = initialize_distributed_training(model_parallel_size)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert model_parallel_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    # Check if we can use bf16 or half precision for training (mixed precision)
    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    
    # Wrap the model with DDP
    model = DDP(model.cuda(local_rank), device_ids=[local_rank])

    print(f"Loaded in {time.time() - start_time:.2f} seconds")

    return Llama(model, tokenizer)


# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================
# ===================================================================================



from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train(model, tokenizer, train_data, batch_size=32, epochs=3):
    # Create the dataset and sampler for distributed training
    dataset = MyDataset(train_data)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Use Adam optimizer
    optimizer = Adam(model.parameters(), lr=5e-5)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensure each epoch gets a different shuffling across processes
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass (modify based on your model's input format)
            input_data = tokenizer.encode(batch)
            outputs = model(input_data)
            
            loss = compute_loss(outputs)  # Define your loss function
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} finished.")

# Example data
train_data = ["Example prompt 1", "Example prompt 2", "Example prompt 3"]

# Initialize and train the model
llama = Llama.build(ckpt_dir="path_to_checkpoints", tokenizer_path="tokenizer.json", max_seq_len=128, max_batch_size=32)
train(llama.model, llama.tokenizer, train_data)
