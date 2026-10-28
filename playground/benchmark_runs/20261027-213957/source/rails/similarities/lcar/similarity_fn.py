"""LCAR: latent preparation plus one joint actual-ad memory read."""

import math
from dataclasses import replace
import weakref

import torch
from torch import nn

from rails.similarities.module import SimilarityModule
from .attention import masked_softmax
from .encoding import backbone_version, encode_latent_history
from .sparse_coding import NonnegativeSparseCoding
from .states import ItemState,JointMemory,ModelContext,UserMemory


class LCARSimilarity(SimilarityModule):
    requires_user_memory=True
    scorer_type='LCAR'
    exact_top_k_method='LearnedBruteForceTopK'

    SOURCE_NAMES = ('recent', 'global', 'selected', 'summary')

    @torch.inference_mode(False)
    def __init__(self,query_embedding_dim,item_embedding_dim,num_item_experts=4,
                 dictionary_size=128,max_active_latents=16,events_per_slot=16,
                 recent_events=32,attention_dim=32,value_dim=32,memory_dim=32,
                 hidden_dim=64,sparse_coding_steps=8,sparse_l1_weight=.01,
                 latent_chunk_size=128,eps=1e-8,use_selected_tokens=True,use_summary_tokens=True,
                 use_global_token=True,use_source_embeddings=True):
        super().__init__()
        if min(query_embedding_dim,item_embedding_dim,num_item_experts,events_per_slot,
               attention_dim,value_dim,memory_dim,hidden_dim,latent_chunk_size)<1 or recent_events<0:
            raise ValueError('Invalid LCAR dimensions or budgets')
        self.query_embedding_dim=query_embedding_dim
        self.item_embedding_dim=item_embedding_dim
        self.num_item_experts=num_item_experts
        self.dictionary_size=dictionary_size
        self.max_active_latents=max_active_latents
        self.events_per_slot=events_per_slot
        self.recent_events=recent_events
        self.attention_dim=attention_dim
        self.value_dim=value_dim
        self.memory_dim=memory_dim
        self.hidden_dim=hidden_dim
        self.latent_chunk_size=latent_chunk_size
        self.eps=eps
        self.use_selected_tokens=use_selected_tokens
        self.use_summary_tokens=use_summary_tokens
        self.use_global_token=use_global_token
        self.use_source_embeddings=use_source_embeddings
        self.sparse_coder=NonnegativeSparseCoding(num_item_experts,dictionary_size,attention_dim,
                                                 max_active_latents,sparse_coding_steps,sparse_l1_weight,eps)
        self.token_projections=nn.ModuleList([nn.Linear(attention_dim,query_embedding_dim)
                                              for _ in range(num_item_experts)])
        self.summary_projections=nn.ModuleList([nn.Linear(query_embedding_dim,memory_dim)
                                                for _ in range(num_item_experts)])
        self.raw_projection=nn.Linear(query_embedding_dim,memory_dim)
        self.global_projection=nn.Linear(query_embedding_dim,memory_dim)
        self.source_embeddings=nn.Parameter(torch.zeros(len(self.SOURCE_NAMES),memory_dim))
        self.memory_key=nn.Linear(memory_dim,attention_dim)
        self.memory_value=nn.Linear(memory_dim,value_dim)
        self.item_experts=nn.ModuleList([nn.Sequential(nn.Linear(item_embedding_dim,hidden_dim),nn.SiLU(),
                                                      nn.Linear(hidden_dim,attention_dim))
                                         for _ in range(num_item_experts)])
        self.query_projection=nn.Linear(attention_dim,attention_dim)
        self.value_query=nn.Linear(attention_dim,value_dim)
        self.expert_gate=nn.Linear(item_embedding_dim,num_item_experts)
        self.expert_scorers=nn.ModuleList([nn.Sequential(nn.Linear(attention_dim+2*value_dim,hidden_dim),
                                                        nn.SiLU(),nn.Linear(hidden_dim,1))
                                           for _ in range(num_item_experts)])
        self._model_cache=None

    def get_extra_state(self):
        return dict(format_version=4,scorer_type='LCAR',query_embedding_dim=self.query_embedding_dim,
                    item_embedding_dim=self.item_embedding_dim,num_item_experts=self.num_item_experts,
                    dictionary_size=self.dictionary_size,max_active_latents=self.max_active_latents,
                    events_per_slot=self.events_per_slot,recent_events=self.recent_events,
                    attention_dim=self.attention_dim,value_dim=self.value_dim,memory_dim=self.memory_dim,
                    hidden_dim=self.hidden_dim,sparse_coding_steps=self.sparse_coder.steps,
                    use_selected_tokens=self.use_selected_tokens,use_summary_tokens=self.use_summary_tokens,
                    use_global_token=self.use_global_token,use_source_embeddings=self.use_source_embeddings,
                    memory_sources=self.SOURCE_NAMES,
                    sparse_l1_weight=self.sparse_coder.l1_weight,latent_chunk_size=self.latent_chunk_size,
                    selection='first_block_native_coefficients_mean_heads',overlap='retain_recent_selected',
                    target_position='prefix_length',target_time='request_or_last_valid',
                    native_normalization='host_reference_capacity',expert_gating='final_logits_only',
                    coefficient_use='support_and_auxiliary_only',eps=self.eps)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():
            raise ValueError('LCAR checkpoint architecture/settings differ; version 4 is required')
        self._model_cache=None
        self.sparse_coder._step_cache=None

    def version(self):
        if any(p.is_inference() for p in self.parameters()):
            raise ValueError('Construct and move LCAR parameters outside inference_mode')
        return (id(self),self.training,tuple((id(p),p._version,p.dtype,p.device) for p in self.parameters()))

    def wrap_encoder(self, backbone, *, adapter=None):
        """Attach LCAR without modifying or registering its parameters in the host."""
        from .encoder import LCAREncoder
        return LCAREncoder(backbone, self, adapter=adapter)

    @torch.no_grad()
    def post_optimizer_step(self):
        self.sparse_coder.project_dictionary_()
        self._model_cache=None

    def prepare_model(self):
        version=self.version()
        cacheable=not self.training and not torch.is_grad_enabled()
        if cacheable and self._model_cache is not None and self._model_cache[0]==version:
            return self._model_cache[1]
        self.sparse_coder.step_size()
        tokens=torch.stack([projection(self.sparse_coder.dictionary[e])
                            for e,projection in enumerate(self.token_projections)])
        sources = self.source_embeddings if self.use_source_embeddings else torch.zeros_like(self.source_embeddings)
        # Source offsets use projection weights only: cached base K/V already include bias.
        source_keys = torch.nn.functional.linear(sources, self.memory_key.weight).float()
        source_values = torch.nn.functional.linear(sources, self.memory_value.weight).float()
        context=ModelContext(tokens*math.sqrt(self.query_embedding_dim),source_keys,source_values,version)
        if cacheable: self._model_cache=version,context
        return context

    def encode_history(self,backbone,lengths,ids,embeddings,payloads,**kwargs):
        """Plug-in entry point; an explicit adapter can support another host model."""
        return encode_latent_history(self,backbone,lengths,ids,embeddings,payloads,**kwargs)

    def prepare_user(self,backbone,lengths,ids,embeddings,payloads,**kwargs):
        """Standalone API; the host need not inherit the repository model bridge."""
        encoded=self.encode_history(backbone,lengths,ids,embeddings,payloads,**kwargs)
        memory=self.finish_user_memory(encoded,event_payloads=dict(payloads,item_ids=ids))
        return replace(memory,encoder_ref=weakref.ref(backbone),encoder_version=backbone_version(backbone))

    def prepare_items(self,item_embeddings,item_ids=None,**kwargs):
        if item_embeddings.ndim!=3:
            raise ValueError('LCAR items must be [1/batch,candidates,features]')
        x=item_embeddings.to(self.expert_gate.weight.dtype)
        queries=torch.stack([expert(x) for expert in self.item_experts],-2)
        indices,mask,coefficients,alpha,fallback,loss=self.sparse_coder(queries,compute_loss=self.training)
        if item_ids is None:
            item_ids=torch.arange(x.shape[1],device=x.device).expand(x.shape[0],-1)
        if item_ids.shape!=x.shape[:2]: raise ValueError('Item IDs and features are misaligned')
        return ItemState(item_ids.clone(),queries,self.query_projection(queries).float(),
                         self.value_query(queries).float(),indices,mask,coefficients,alpha,
                         self.expert_gate(x).float().softmax(-1),fallback,loss,self.version())

    def finish_user_memory(self,encoded,*,event_payloads=None):
        """Pack originals, the native history readout, and individual latent summaries."""
        if encoded.context.version!=self.version(): raise ValueError('Stale LCAR encoder output')
        raw,mask=encoded.original,encoded.mask
        batch,length,_=raw.shape
        if encoded.latent_states.shape[1]!=self.num_item_experts*self.dictionary_size:
            raise ValueError('User preparation requires every LCAR latent, not a target-selected subset')
        if length==0:
            raw=raw.new_zeros(batch,1,raw.shape[-1])
            mask=torch.zeros(batch,1,device=raw.device,dtype=torch.bool)
        length=raw.shape[1]
        counts=mask.sum(-1)
        slots=encoded.selected_positions.reshape(batch,self.num_item_experts,self.dictionary_size,-1)
        slot_mask=encoded.selected_mask.reshape_as(slots)
        positions=torch.arange(length,device=raw.device).expand(batch,-1)
        recent_valid=mask & (mask.long().cumsum(-1)>counts[:,None]-self.recent_events)
        recent=positions.masked_fill(~recent_valid,length).argsort(dim=-1,stable=True)[:,:min(self.recent_events,length)]
        recent_mask=recent_valid.gather(1,recent)
        refs=torch.cat([slots.reshape(batch,-1),recent],-1)
        present=torch.cat([slot_mask.reshape(batch,-1),recent_mask],-1)
        retention=torch.zeros(batch,length,device=raw.device,dtype=torch.long)
        retention.scatter_add_(1,refs,present.long())
        retained=retention>0
        packed_size=max(1,int(retained.sum(-1).max().item()))
        order=(~retained).long().argsort(dim=-1,stable=True)[:,:packed_size]
        packed_mask=retained.gather(1,order)&mask.gather(1,order)
        reverse=torch.zeros_like(positions)
        reverse.scatter_(1,order,torch.arange(packed_size,device=raw.device).expand(batch,-1))
        packed_raw=raw.gather(1,order[...,None].expand(-1,-1,raw.shape[-1]))
        tokens=self.raw_projection(packed_raw.to(self.raw_projection.weight.dtype)).float()
        latent=encoded.latent_states.reshape(batch,self.num_item_experts,self.dictionary_size,-1)
        summaries=torch.stack([projection(latent[:,e].to(projection.weight.dtype))
                               for e,projection in enumerate(self.summary_projections)],1).float()
        summary_mask=(counts>0)[:,None,None].expand(batch,self.num_item_experts,self.dictionary_size)
        summaries=summaries*summary_mask[...,None]
        # Use original valid positions, not a padded final slot or the latent suffix.
        last_position = positions.masked_fill(~mask, -1).amax(-1).clamp_min(0)
        if encoded.history_states.shape[1]:
            last_history = encoded.history_states[torch.arange(batch, device=raw.device), last_position]
        else:
            last_history = raw.new_zeros(batch, self.query_embedding_dim)
        last_history = last_history * (counts > 0).unsqueeze(-1)
        global_mask = (counts > 0).unsqueeze(-1) & self.use_global_token
        global_token = self.global_projection(last_history.to(self.global_projection.weight.dtype))[:,None].float()
        global_token = global_token * global_mask.unsqueeze(-1)
        payloads={}
        for name,tensor in dict(event_payloads or {}).items():
            if isinstance(tensor,torch.Tensor) and tensor.ndim>=2 and tensor.shape[:2]==mask.shape:
                index=order.reshape(batch,packed_size,*([1]*(tensor.ndim-2))).expand(
                    batch,packed_size,*tensor.shape[2:])
                payloads[name]=tensor.gather(1,index)
        def kv(t):
            return (self.memory_key(t.to(self.memory_key.weight.dtype)).float(),
                    self.memory_value(t.to(self.memory_value.weight.dtype)).float())
        keys,values=kv(tokens)
        summary_keys,summary_values=kv(summaries)
        global_keys,global_values=kv(global_token)
        return UserMemory(
            tokens=tokens,keys=keys,values=values,positions=order.masked_fill(~packed_mask,-1),
            retained_mask=packed_mask,lengths=counts,recent_indices=reverse.gather(1,recent),
            recent_mask=recent_mask,
            selected_indices=reverse.gather(1,slots.reshape(batch,-1)).reshape_as(slots),
            selected_mask=slot_mask,summaries=summaries,summary_keys=summary_keys,
            summary_values=summary_values,summary_mask=summary_mask,
            last_history_state=last_history,global_token=global_token,global_keys=global_keys,
            global_values=global_values,global_mask=global_mask,event_payloads=payloads,
            context=encoded.context,version=self.version())

    def _check_prepared(self,memory,items):
        if memory.encoder_ref is not None:
            encoder=memory.encoder_ref()
            if encoder is None or backbone_version(encoder)!=memory.encoder_version:
                raise ValueError('Stale history encoder state; rebuild LCAR user_memory')
        if memory.version!=self.version() or items.version!=self.version():
            raise ValueError('Stale or foreign LCAR prepared state')
        if items.queries.shape[0] not in (1,memory.batch_size):
            raise ValueError('LCAR candidates must be shared or aligned with users')

    def gather_memory(self,memory,items):
        """Concatenate originals and individual summaries; preserve recent overlap."""
        batch,n=memory.batch_size,items.num_candidates
        active=items.active_indices.expand(batch,-1,-1,-1)
        active_mask=items.active_mask.expand_as(active)
        rows=torch.arange(batch,device=active.device)[:,None,None,None]
        experts=torch.arange(self.num_item_experts,device=active.device)[None,None,:,None]
        selections=memory.selected_indices[rows,experts,active]
        valid=memory.selected_mask[rows,experts,active]&active_mask[...,None]
        sentinel=memory.tokens.shape[1]
        # Flatten expert, active-latent and selected-event axes into ONE union.
        union=selections.masked_fill(~valid,sentinel).flatten(-3).sort(-1).values
        unique=torch.cat([torch.ones_like(union[...,:1],dtype=torch.bool),union[...,1:]!=union[...,:-1]],-1)
        union_mask=(union<sentinel)&unique&self.use_selected_tokens
        union_size=int(union_mask.sum(-1).max().item()) if batch and n else 0
        # Compact duplicates before materializing candidate memory K/V.
        compact=(~union_mask).long().argsort(dim=-1,stable=True)[...,:union_size]
        union=union.gather(-1,compact).clamp_max(sentinel-1)
        union_mask=union_mask.gather(-1,compact)
        row=torch.arange(batch,device=active.device)[:,None,None]
        recent=memory.recent_indices[:,None].expand(batch,n,-1)
        recent_mask=memory.recent_mask[:,None].expand_as(recent)
        summary_mask=memory.summary_mask[rows,experts,active]&active_mask&self.use_summary_tokens
        global_mask = memory.global_mask[:,None].expand(batch,n,-1)
        sizes = (recent.shape[-1], 1, union.shape[-1], active.shape[-2]*active.shape[-1])
        source_ids = torch.cat([torch.full((size,),source,device=active.device,dtype=torch.long)
                                for source,size in enumerate(sizes)])

        def gather(raw,global_values,summary,offsets):
            base = torch.cat([
                raw[row,recent],global_values[:,None].expand(batch,n,-1,-1),
                raw[row,union],summary[rows,experts,active].flatten(-3,-2)],-2)
            return base + offsets[source_ids]

        keys=gather(memory.keys,memory.global_keys,memory.summary_keys,memory.context.source_keys)
        values=gather(memory.values,memory.global_values,memory.summary_values,memory.context.source_values)
        mask=torch.cat([recent_mask,global_mask,union_mask,summary_mask.flatten(-2)],-1)
        return JointMemory(keys,values,mask,memory.positions[row,union],union_mask,source_ids)

    def attention_source_mass(self, weights, joint_memory):
        """Detached diagnostics from an existing read, without another attention pass."""
        return {name: weights.detach()[...,joint_memory.source_ids == source].sum(-1).mean()
                for source,name in enumerate(self.SOURCE_NAMES)}

    def attend_prepared(self,memory,items,return_weights=False):
        self._check_prepared(memory,items)
        joint=self.gather_memory(memory,items)
        q=items.attention_queries.expand(memory.batch_size,-1,-1,-1)
        logits=torch.einsum('bned,bntd->bnet',q,joint.keys)/math.sqrt(self.attention_dim)
        weights=masked_softmax(logits,joint.mask.unsqueeze(-2))
        outputs=torch.einsum('bnet,bntv->bnev',weights,joint.values)
        return (outputs,weights,joint) if return_weights else outputs

    def expert_logits(self,outputs,items):
        q=items.queries.expand(outputs.shape[0],-1,-1,-1).float()
        vq=items.value_queries.expand(outputs.shape[0],-1,-1,-1)
        features=torch.cat([q,outputs,vq*outputs],-1)
        return torch.stack([scorer(features[...,e,:].to(scorer[0].weight.dtype)).squeeze(-1).float()
                            for e,scorer in enumerate(self.expert_scorers)],-1)

    def score_prepared(self,user_memory,item_state,return_auxiliary_losses=True,**kwargs):
        outputs=self.attend_prepared(user_memory,item_state)
        logits=self.expert_logits(outputs,item_state)
        scores=(logits*item_state.expert_weights).sum(-1)
        aux=item_state.auxiliary_losses() if self.training and return_auxiliary_losses else {}
        return scores,aux

    def forward(self,query_embeddings,item_embeddings,**kwargs):
        memory=kwargs.get('user_memory')
        if memory is None or query_embeddings.shape[0]!=memory.batch_size:
            raise ValueError('LCAR requires explicit aligned user_memory')
        items=kwargs.get('item_state')
        if items is None: items=self.prepare_items(item_embeddings,kwargs.get('item_ids'))
        return self.score_prepared(memory,items)
