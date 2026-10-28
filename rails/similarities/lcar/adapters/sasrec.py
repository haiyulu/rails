"""SASRec native equations with independent virtual-candidate reads."""

import torch
from torch.nn import functional as F

from .base import BaseAdapter
from ..attention import masked_softmax

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


