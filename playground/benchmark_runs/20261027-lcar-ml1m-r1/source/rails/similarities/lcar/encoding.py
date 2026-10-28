"""Backbone-independent orchestration of shared history and virtual candidates."""

import torch
from torch.utils.checkpoint import checkpoint

from .backbones import get_backbone_adapter
from .states import EncoderOutput, HistoryCache


def _same(a,b):
    return a is None and b is None or (a is not None and b is not None and torch.equal(a,b))


def backbone_version(backbone):
    if hasattr(backbone,'history_version'):
        return backbone.history_version()
    return (id(backbone),backbone.training,
            tuple((id(p),p._version,p.dtype,p.device) for p in backbone.parameters()))


def encode_latent_history(module,backbone,lengths,ids,embeddings,payloads,
                          adapter=None,history_cache=None,token_indices=None):
    """Run all virtual candidates, or a subset for reference/chunking validation.

    The adapter is a parameter-free view of the host blocks. LCAR never registers
    the host as a child or copies its block parameters.
    """
    adapter=get_backbone_adapter(backbone) if adapter is None else adapter
    original,mask=adapter.preprocess(lengths,ids,embeddings,payloads)
    context=module.prepare_model()
    bases=context.initial_tokens.flatten(0,1)
    if token_indices is not None:
        bases=bases[token_indices]
    positions=adapter.target_positions(lengths)
    latents=bases[None].expand(original.shape[0],-1,-1)+positions[:,None]
    timestamps=payloads.get('timestamps')
    if timestamps is not None:
        timestamps=timestamps[:,:mask.shape[1]].masked_fill(~mask,0)
    target_times=payloads.get('request_timestamps')
    version=backbone_version(backbone)
    if history_cache is not None:
        if torch.is_grad_enabled() or backbone.training:
            raise ValueError('LCAR history caches are for evaluation without gradients')
        if not isinstance(history_cache,HistoryCache) or history_cache.version!=version:
            raise ValueError('Stale or foreign LCAR history cache')
        if len(history_cache.layers)!=adapter.num_layers:
            raise ValueError('Build LCAR history caches in evaluation without gradients')
        if not (torch.equal(history_cache.original,original) and torch.equal(history_cache.mask,mask)
                and _same(history_cache.timestamps,timestamps) and _same(history_cache.target_times,target_times)):
            # A changed prefix needs fresh history; latent reads always start from layer zero.
            history_cache=None
    history=original
    layers=[]
    selected=selected_mask=None
    for index in range(adapter.num_layers):
        cached_layer=history_cache.layers[index] if history_cache is not None else None

        def step(h,z,index=index,cached_layer=cached_layer):
            if cached_layer is None:
                new_history,layer_context=adapter.history_step(index,h,mask,lengths,payloads)
            else:
                new_history,layer_context=h,cached_layer
            outputs=[]
            selections=[]
            selection_masks=[]
            for start in range(0,z.shape[1],module.latent_chunk_size):
                updated,coefficients=adapter.latent_step(index,z[:,start:start+module.latent_chunk_size],layer_context)
                outputs.append(updated)
                if index==0:
                    score=coefficients.float().mean(1).masked_fill(~mask[:,None,:],-torch.inf)
                    chosen=score.argsort(dim=-1,descending=True,stable=True)[...,:min(module.events_per_slot,mask.shape[1])]
                    selections.append(chosen)
                    selection_masks.append(mask[:,None,:].expand_as(score).gather(-1,chosen))
            empty=torch.empty(z.shape[0],0,0,device=z.device,dtype=torch.long)
            return (new_history,torch.cat(outputs,1),
                    torch.cat(selections,1) if selections else empty,
                    torch.cat(selection_masks,1) if selection_masks else empty.bool())

        # History cache construction is kept outside checkpointed recomputation.
        # Its tensors stay attached during training but are reused only in evaluation.
        if cached_layer is None and not torch.is_grad_enabled():
            new_history,layer_context=adapter.history_step(index,history,mask,lengths,payloads)
            cached_layer=layer_context
            layers.append(layer_context)
            history=new_history
            # Use the just-built context once without re-projecting history.
            h,z,chosen,chosen_mask=step(history,latents,index=index,cached_layer=layer_context)
        else:
            if adapter.activation_checkpoint and torch.is_grad_enabled():
                h,z,chosen,chosen_mask=checkpoint(step,history,latents,use_reentrant=False)
            else:
                h,z,chosen,chosen_mask=step(history,latents)
            if history_cache is not None:
                layers.append(cached_layer)
        history,latents=h,z
        if index==0:
            selected,selected_mask=chosen,chosen_mask
    final_history=(history_cache.final_states if history_cache is not None else adapter.postprocess(history))
    final_latents=adapter.postprocess(latents)
    # Request buffers are often reused in place; cache metadata must be a snapshot.
    cached_target_times=target_times.clone() if target_times is not None else None
    cache=HistoryCache(original,mask,cached_target_times,timestamps,tuple(layers),final_history,version)
    return EncoderOutput(original,mask,final_history,final_latents,selected,selected_mask,context,cache)
