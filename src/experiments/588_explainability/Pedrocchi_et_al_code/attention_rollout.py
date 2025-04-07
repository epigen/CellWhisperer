import torch

## Code to calculate attention rollout per batch
def attention_rollout(args, weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    length = weights.shape[2]
    I = (torch.eye(length) * 0.5).to(device)
    rollouts = []
    for cell in range(0, weights.shape[1]):
        rollout = torch.eye(length).to(device)
        for layer in range(0, args.nlayers):
            rollout = torch.matmul(0.5*weights[layer, cell] + I, rollout)
        rollouts.append(rollout[0])
    return torch.stack(rollouts)


## Code to make modules return attention weights
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True

        return forward_orig(*args, **kwargs)

    m.forward = wrap

## Code to save attention weights
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


## Initialize model and dataloader
model = load_model(args)
model.eval()
dataloader = load_dataloader(args)
save_output = SaveOutput()
final_weights = []

## Register hooks for all attention modules
for module in model.modules():
    if isinstance(module, torch.nn.MultiheadAttention):
        patch_attention(module)
        module.register_forward_hook(save_output)

            
for batch in dataloader:
    with torch.no_grad():
        model(batch)
    attention_weights = torch.stack(save_output.outputs)
    save_output.clear()
    final_weights.append(attention_rollout(args, attention_weights))
    