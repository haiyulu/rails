"""Native-block adapters. All SASRec/HSTU-specific knowledge lives here.

An adapter owns no parameters: it reads the host's existing blocks. Register an
adapter for another backbone without changing LCAR's dictionary or scorer.
"""

from typing import Protocol

import torch
from torch.nn import functional as F


def masked_softmax(logits, mask):
    logits=logits.masked_fill(~mask, -torch.inf)
    maximum=logits.amax(-1, keepdim=True)
    maximum=torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    weights=(logits-maximum).exp()
    return weights/weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class BackboneAdapter(Protocol):
    num_layers: int
    activation_checkpoint: bool
    def preprocess(self, lengths, ids, embeddings, payloads): ...
    def target_positions(self, lengths): ...
    def history_step(self, index, history, mask, lengths, payloads): ...
    def latent_step(self, index, latents, context): ...
    def postprocess(self, states): ...


_ADAPTERS={}


def register_backbone_adapter(backbone_type, adapter_factory):
    """Register a host class (or its name) and a parameter-free adapter factory."""
    _ADAPTERS[backbone_type]=adapter_factory


def get_backbone_adapter(backbone):
    for cls in type(backbone).__mro__:
        factory=_ADAPTERS.get(cls, _ADAPTERS.get(cls.__name__))
        if factory is not None:
            return factory(backbone)
    raise ValueError(f'No LCAR adapter registered for {type(backbone).__name__}')


class BaseAdapter:
    def __init__(self, backbone):
        self.backbone=backbone
        self.num_layers=backbone._num_blocks
        self.activation_checkpoint=getattr(backbone, '_activation_checkpoint', False)

    def preprocess(self, lengths, ids, embeddings, payloads):
        new_lengths, raw, valid=self.backbone._input_features_preproc(lengths,ids,embeddings,payloads)
        if raw.shape[:2]!=ids.shape or not torch.equal(lengths,new_lengths):
            raise ValueError('LCAR requires one original token per event; register an adapter for expanded inputs')
        mask=valid.squeeze(-1).bool() & (torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None])
        return raw*mask.unsqueeze(-1),mask

    def target_positions(self, lengths):
        positions=getattr(self.backbone._input_features_preproc, '_pos_emb', None)
        if positions is None:
            raise ValueError('Register a backbone adapter with target-position initialization')
        if bool((lengths>=positions.num_embeddings).any()):
            raise ValueError('LCAR requires a reserved next-candidate position')
        return positions(lengths)

    def postprocess(self, states):
        return self.backbone._output_postproc(states)


class SASRecAdapter(BaseAdapter):
    def _project(self,index,states):
        layer=self.backbone.attention_layers[index]
        if layer.bias_k is not None or layer.add_zero_attn or not layer._qkv_same_embed_dim:
            raise ValueError('LCAR SASRec adapter requires ordinary self-attention projections')
        width=states.shape[-1]
        q_input=F.layer_norm(states,(width,),eps=1e-8)
        wq,wk,wv=layer.in_proj_weight.chunk(3)
        bq,bk,bv=layer.in_proj_bias.chunk(3) if layer.in_proj_bias is not None else (None,None,None)
        def split(t): return t.reshape(*t.shape[:-1],layer.num_heads,width//layer.num_heads)
        return q_input,split(F.linear(q_input,wq,bq)),split(F.linear(states,wk,bk)),split(F.linear(states,wv,bv))

    def _finish(self,index,q_input,attention):
        layer=self.backbone.attention_layers[index]
        output=layer.out_proj(attention.flatten(-2).to(layer.out_proj.weight.dtype))
        return self.backbone.forward_layers[index](F.layer_norm(q_input+output,(q_input.shape[-1],),eps=1e-8))

    def history_step(self,index,history,mask,lengths,payloads):
        q_input,q,k,v=self._project(index,history)
        length=history.shape[1]
        causal=torch.ones(length,length,device=history.device,dtype=torch.bool).tril()
        allowed=causal[None,None] & mask[:,None,None,:]
        logits=torch.einsum('bthd,bshd->bhts',q.float(),k.float())/(q.shape[-1]**.5)
        coefficients=masked_softmax(logits,allowed)
        layer=self.backbone.attention_layers[index]
        weights=F.dropout(coefficients,layer.dropout,layer.training)
        context=torch.einsum('bhts,bshd->bthd',weights,v.float())
        output=self._finish(index,q_input,context)*mask.unsqueeze(-1)
        return output,dict(k=k,v=v,mask=mask)

    def latent_step(self,index,latents,context):
        q_input,q,k,v=self._project(index,latents)
        history_logits=torch.einsum('bjhd,bthd->bhjt',q.float(),context['k'].float())/(q.shape[-1]**.5)
        self_logits=(q.float()*k.float()).sum(-1).permute(0,2,1)/(q.shape[-1]**.5)
        logits=torch.cat([history_logits,self_logits.unsqueeze(-1)],-1)
        mask=context['mask'][:,None,None,:].expand_as(history_logits)
        mask=torch.cat([mask,torch.ones_like(self_logits.unsqueeze(-1),dtype=torch.bool)],-1)
        coefficients=masked_softmax(logits,mask)
        layer=self.backbone.attention_layers[index]
        weights=F.dropout(coefficients,layer.dropout,layer.training)
        output=(torch.einsum('bhjt,bthd->bjhd',weights[...,:-1],context['v'].float())
                + weights[...,-1].permute(0,2,1).unsqueeze(-1)*v.float())
        return self._finish(index,q_input,output),coefficients[...,:-1]


class HSTUAdapter(BaseAdapter):
    """Exact native pointwise HSTU equations, with independent target reads.

    Normalization uses the host's fixed reference capacity, including reserved
    output positions. It never depends on the number or chunking of latents.
    """
    def __init__(self,backbone):
        super().__init__(backbone)
        self.layers=backbone._hstu._attention_layers
        self.reference_length=backbone._attn_mask.shape[-1]
        if any(layer._normalization not in ('rel_bias','hstu_rel_bias') for layer in self.layers):
            raise ValueError('LCAR HSTU adapter supports native rel_bias/hstu_rel_bias attention')

    def _project(self,index,states):
        layer=self.layers[index]
        projection=F.linear(layer._norm_input(states),layer._uvqk.T)
        if layer._linear_activation=='silu': projection=F.silu(projection)
        elif layer._linear_activation!='none': raise ValueError('Unsupported HSTU activation')
        hv=layer._linear_dim*layer._num_heads
        hq=layer._attention_dim*layer._num_heads
        u,v,q,k=projection.split([hv,hv,hq,hq],dim=-1)
        def split(t): return t.reshape(*t.shape[:-1],layer._num_heads,-1)
        return u,split(v),split(q),split(k)

    def _finish(self,index,states,u,attention):
        layer=self.layers[index]
        a=layer._norm_attn_output(attention.flatten(-2).to(u.dtype))
        inputs=torch.cat([u,a,u*a],-1) if layer._concat_ua else u*a
        return layer._o(F.dropout(inputs,layer._dropout_ratio,layer.training))+states

    def _bias(self,index,mask,lengths,payloads):
        layer=self.layers[index]
        module=layer._rel_attn_bias
        timestamps=payloads.get('timestamps')
        if timestamps is None or module is None:
            return None,None,None
        batch,length=mask.shape
        pos=torch.arange(length,device=mask.device).expand(batch,-1)
        last=(lengths-1).clamp_min(0)
        times=timestamps[:,:length]
        request=payloads.get('request_timestamps')
        if request is None: request=times.gather(1,last[:,None]).squeeze(1)
        request=request.reshape(batch)
        request=torch.where(lengths>0,request,torch.zeros_like(request))
        safe=torch.where(mask,times,request[:,None])
        next_times=torch.cat([safe[:,1:],request[:,None]],1)
        next_times=torch.where(pos+1<lengths[:,None],next_times,request[:,None])
        weights=getattr(module,'_pos_w',getattr(module,'_w',None))
        if weights is None: raise ValueError('Register an adapter for this HSTU relative-bias module')
        center=module._max_seq_len-1
        def bias(query_pos,key_pos,query_time,key_time):
            result=weights[center+key_pos-query_pos]
            if hasattr(module,'_ts_w'):
                buckets=module._bucketization_fn(query_time-key_time).clamp(0,module._num_buckets).detach()
                result=result+module._ts_w[buckets]
            return result
        history=bias(pos[:,:,None],pos[:,None,:],next_times[:,:,None],safe[:,None,:])
        target=bias(lengths[:,None],pos,request[:,None],safe)
        self_bias=bias(lengths,lengths,request,request)
        return history,target,self_bias

    def history_step(self,index,history,mask,lengths,payloads):
        u,v,q,k=self._project(index,history)
        history_bias,target_bias,self_bias=self._bias(index,mask,lengths,payloads)
        logits=torch.einsum('bthd,bshd->bhts',q.float(),k.float())
        if history_bias is not None: logits=logits+history_bias[:,None]
        coefficients=F.silu(logits)/self.reference_length
        length=history.shape[1]
        causal=torch.ones(length,length,device=history.device,dtype=torch.bool).tril()
        coefficients=coefficients*(causal[None,None]&mask[:,None,None,:])
        attention=torch.einsum('bhts,bshv->bthv',coefficients,v.float())
        output=self._finish(index,history,u,attention)*mask.unsqueeze(-1)
        return output,dict(k=k,v=v,mask=mask,target_bias=target_bias,self_bias=self_bias)

    def latent_step(self,index,latents,context):
        u,v,q,k=self._project(index,latents)
        logits=torch.einsum('bjhd,bthd->bhjt',q.float(),context['k'].float())
        self_logits=(q.float()*k.float()).sum(-1).permute(0,2,1)
        if context['target_bias'] is not None:
            logits=logits+context['target_bias'][:,None,None,:]
            self_logits=self_logits+context['self_bias'][:,None,None]
        coefficients=F.silu(logits)/self.reference_length
        coefficients=coefficients*context['mask'][:,None,None,:]
        self_coeff=F.silu(self_logits)/self.reference_length
        output=(torch.einsum('bhjt,bthv->bjhv',coefficients,context['v'].float())
                +self_coeff.permute(0,2,1).unsqueeze(-1)*v.float())
        return self._finish(index,latents,u,output),coefficients


register_backbone_adapter('SASRec',SASRecAdapter)
register_backbone_adapter('HSTU',HSTUAdapter)
