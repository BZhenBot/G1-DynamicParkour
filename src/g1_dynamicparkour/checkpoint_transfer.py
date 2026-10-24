"""Verified rsl_rl 5.4.2 MLP input expansion; derived weights are not newly trained skills."""
import copy
import torch

def expand_model_state(source,new_input_dim):
    result=copy.deepcopy(source)
    old_weight=source["mlp.0.weight"]
    old_input_dim=old_weight.shape[1]
    if new_input_dim<old_input_dim:
        raise ValueError("Cannot shrink or reorder observation prefixes")
    if source["obs_normalizer._mean"].shape != (1,old_input_dim):
        raise ValueError("Unsupported normalizer shape; inspect actual checkpoint first")
    weight=torch.zeros((old_weight.shape[0],new_input_dim),device=old_weight.device,dtype=old_weight.dtype)
    weight[:,:old_input_dim]=old_weight
    result["mlp.0.weight"]=weight
    for name,initial in (("_mean",0.),("_var",1.),("_std",1.)):
        key="obs_normalizer."+name
        extended=torch.full((1,new_input_dim),initial,device=source[key].device,dtype=source[key].dtype)
        extended[:,:old_input_dim]=source[key]
        result[key]=extended
    # Keep the original count/prefix statistics; optimizer must be initialized fresh.
    return result