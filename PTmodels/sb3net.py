import torch
import json 
import pickle
from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn as nn
from map_tool_box.scripts.tensorrtConversion.common import elementwise_mode3


# elements should be float32 tensors cast to same device as model
# img should be normalized between 0 and 1
# vec should be normalized between -1 and 1
# .detach().cpu().numpy() to be done outside 
class SB3Net(torch.nn.Module):
    def __init__(self, cnn_extractor, linear_extractor, vec_extractor, q_net):
        super(SB3Net, self).__init__()
        self.cnn_extractor = cnn_extractor
        self.linear_extractor = linear_extractor
        self.vec_extractor = vec_extractor
        self.q_net = q_net
    
    def forward(self, img, vec):
        img = self.cnn_extractor(img)
        img = self.linear_extractor(img)
        vec = self.vec_extractor(vec)
        cat = torch.cat([img, vec], dim=1)
        pred = self.q_net(cat)
        action = torch.argmax(pred, axis=1).int()
        return action

class TMRModule(torch.nn.Module):
    def __init__(self, base_module: torch.nn.Module, tol: float = 0.0):
        super().__init__()
        self.base = base_module
        self.tol = float(tol)

    def forward(self, *inputs, **kwargs):
        y1 = self.base(*inputs, **kwargs)
        y2 = self.base(*inputs, **kwargs)
        y3 = self.base(*inputs, **kwargs)
        return self._mode_outputs(y1, y2, y3)

    def _mode_outputs(self, y1, y2, y3):
        return elementwise_mode3(y1, y2, y3, tol=self.tol)

def pick_layer_by_idx_name(model, lyr_idx):
    count = 0
    for mod, name in iter_leaves(model):  # assumes iter_leaves(model) -> (module, full_name)
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,torch.nn.Linear)):
            if count == lyr_idx:
                return mod, name
            count += 1

class FullTMRModule(torch.nn.Module):
    def __init__(self, base_module: torch.nn.Module):
        super().__init__()
        self.base = base_module

    def forward(self, *inputs, **kwargs):
        y1 = self.base(*inputs, **kwargs)
        y2 = self.base(*inputs, **kwargs)
        y3 = self.base(*inputs, **kwargs)
        return self._mode_outputs(y1, y2, y3)

    def _mode_outputs(self, y1, y2, y3):
        return elementwise_mode3(y1, y2, y3)

def pick_layer_by_idx_name(model, lyr_idx):
    count = 0
    for mod, name in iter_leaves(model):  # assumes iter_leaves(model) -> (module, full_name)
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,torch.nn.Linear)):
            if count == lyr_idx:
                return mod, name
            count += 1


def iter_leaves(module: nn.Module, prefix: str = None):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if any(child.named_children()):
            yield from iter_leaves(child, full_name)
        else:
            yield child, full_name

def save_featuremap_shapes(
    model,
    json_path,
    obs, 
    vec
):
    model.eval()

    leaves= list(iter_leaves(model))
    module_to_index= {id(m): i for i, (m, _) in enumerate(leaves)}

    shapes = {}
    hooks = []

    def make_hook(idx: int):
        def hook(_m, inp, _out):
            shapes[str(idx)] = inp[0].shape
        return hook

    for i, (m, _) in enumerate(leaves):
        hooks.append(m.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(obs, vec)

    for h in hooks:
        h.remove()

    with open(json_path, "w") as f:
        json.dump(shapes, f, indent=2)

    return shapes

if __name__ == '__main__':
    mapUT = 'blocks'
    pickle_path = f'./{mapUT}/sb3net.p'
    with open(pickle_path, 'rb') as f:
        model_arch = pickle.load(f)
    model = SB3Net(model_arch.cnn_extractor, model_arch.linear_extractor, model_arch.vec_extractor, model_arch.q_net)
    print(model)

    if mapUT == 'NH':
        obs_shape = (3, 3,144,256) 
        vec_shape = (3,12)
    elif mapUT == 'blocks':
        obs_shape = (1,4,36,64) 
        vec_shape = (1,12)

    dummy_obs = torch.randn(*obs_shape).to('cuda')
    dummy_vec = torch.randn(*vec_shape).to('cuda')

    save_featuremap_shapes(model, f"{mapUT}/embeddings_shape.json", dummy_obs, dummy_vec)