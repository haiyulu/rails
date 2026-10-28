"""LCAR correctness only: no benchmark entry points or timing workloads."""

import io
from dataclasses import replace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rails.similarities.lcar import LCARSimilarity, register_backbone_adapter
from rails.similarities.lcar.backbones import get_backbone_adapter
from rails.indexing.learned_top_k import LearnedBruteForceTopK
from playground.test_expert_history import small_backbone, configured_backbone, ROOT


def lcar(**kwargs):
    options=dict(query_embedding_dim=6,item_embedding_dim=6,dictionary_size=5,
                 max_active_latents=2,events_per_slot=2,recent_events=3,
                 attention_dim=3,value_dim=3,memory_dim=3,hidden_dim=12,latent_chunk_size=3)
    options.update(kwargs)
    return LCARSimilarity(**options)


def fixture(device='cpu'):
    ids=torch.tensor([[1,2,3,4,5,0,0,0,0,0,0],[6,7,8,0,0,0,0,0,0,0,0]],device=device)
    lengths=torch.tensor([5,3],device=device)
    timestamps=(torch.arange(11,device=device)*100+1000)[None].expand(2,-1).clone()
    return lengths,ids,{'timestamps':timestamps,'request_timestamps':torch.tensor([2000,2000],device=device)}


def encode(model,inputs,**kwargs):
    lengths,ids,payloads=inputs
    return model._ndp_module.encode_history(model,lengths,ids,model.get_item_embeddings(ids),payloads,**kwargs)


def native_single_target(model,encoded,row,index,inputs):
    """Independent native block execution with just ONE appended candidate."""
    lengths,ids,payloads=inputs
    length=int(lengths[row])
    target=(encoded.context.initial_tokens.flatten(0,1)[index]
            +model._input_features_preproc._pos_emb(lengths[row]))
    states=torch.cat([encoded.original[row,:length],target[None]],0)[None]
    first_scores=None
    if type(model).__name__=='SASRec':
        mask=torch.ones(length+1,length+1,device=states.device,dtype=torch.bool).triu(1)
        for layer_index,attention in enumerate(model.attention_layers):
            q=F.layer_norm(states,(states.shape[-1],),eps=1e-8)
            output,weights=attention(q,states,states,attn_mask=mask,average_attn_weights=False)
            if layer_index==0: first_scores=weights[0,:,-1,:length].mean(0)
            states=model.forward_layers[layer_index](F.layer_norm(q+output,(states.shape[-1],),eps=1e-8))
        return model._output_postproc(states)[0,-1],first_scores
    import fbgemm_gpu  # Native jagged reference only.
    capacity=model._attn_mask.shape[-1]
    dense=F.pad(states,(0,0,0,capacity-length-1))
    offsets=torch.tensor([0,length+1],device=states.device)
    jagged=torch.ops.fbgemm.dense_to_jagged(dense,[offsets])[0]
    timestamps=None
    if 'timestamps' in payloads:
        request=payloads.get('request_timestamps',payloads['timestamps'][torch.arange(ids.shape[0],device=ids.device),(lengths-1).clamp_min(0)])[row]
        timestamps=request.expand(1,capacity).clone()
        timestamps[:,:length]=payloads['timestamps'][row,:length]
    mask=1.-model._attn_mask.float()
    for layer_index,layer in enumerate(model._hstu._attention_layers):
        if layer_index==0:
            projected=F.linear(layer._norm_input(jagged),layer._uvqk.T)
            if layer._linear_activation=='silu': projected=F.silu(projected)
            hv,hq=layer._num_heads*layer._linear_dim,layer._num_heads*layer._attention_dim
            _,_,q,k=projected.split([hv,hv,hq,hq],-1)
            q=q.reshape(length+1,layer._num_heads,-1)
            k=k.reshape(length+1,layer._num_heads,-1)
            logits=torch.einsum('hd,thd->ht',q[-1],k[:length])
            if timestamps is not None and layer._rel_attn_bias is not None:
                logits=logits+layer._rel_attn_bias(timestamps)[0,length,:length][None]
            first_scores=(F.silu(logits)/capacity).mean(0)
        jagged,_=layer(jagged,offsets,timestamps,mask)
    return model._output_postproc(jagged)[-1],first_scores


class LCARTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def model(self,backbone='SASRec',device='cpu',**kwargs):
        return small_backbone(backbone,device,interaction=lcar(**kwargs)).eval()

    def test_native_sasrec_independent_target_equivalence_and_selection(self):
        import gin
        gin.bind_parameter('sasrec_encoder.num_heads',2)
        try:
            self._native_equivalence('SASRec','cpu')
        finally:
            gin.clear_config()

    @unittest.skipUnless(torch.cuda.is_available(),'Native HSTU reference requires CUDA')
    def test_native_hstu_independent_target_equivalence_and_selection(self):
        import gin
        gin.bind_parameter('hstu_encoder.num_heads',2)
        try:
            self._native_equivalence('HSTU','cuda')
        finally:
            gin.clear_config()

    def _native_equivalence(self,name,device):
        with torch.compiler.set_stance('force_eager'),torch.inference_mode():
            model=self.model(name,device)
            inputs=fixture(device)
            encoded=encode(model,inputs)
            for row in range(2):
                for token in (0,4,7):
                    expected,scores=native_single_target(model,encoded,row,token,inputs)
                    torch.testing.assert_close(encoded.latent_states[row,token],expected,atol=3e-5,rtol=3e-5)
                    chosen=scores.argsort(dim=-1,descending=True,stable=True)[:2]
                    self.assertTrue(torch.equal(encoded.selected_positions[row,token],chosen))

    def test_latent_permutation_chunk_count_and_history_independence(self):
        with torch.inference_mode():
            for name in ('SASRec','HSTU'):
                model=self.model(name)
                module=model._ndp_module
                inputs=fixture()
                baseline=encode(model,inputs)
                permutation=torch.randperm(10)
                other=encode(model,inputs,token_indices=permutation)
                torch.testing.assert_close(other.history_states,baseline.history_states)
                torch.testing.assert_close(other.latent_states,baseline.latent_states[:,permutation],atol=1e-5,rtol=1e-5)
                self.assertTrue(torch.equal(other.selected_positions,baseline.selected_positions[:,permutation]))
                one=encode(model,inputs,token_indices=torch.tensor([3]))
                torch.testing.assert_close(one.latent_states[:,0],baseline.latent_states[:,3],atol=1e-5,rtol=1e-5)
                module.latent_chunk_size=7
                chunked=encode(model,inputs)
                torch.testing.assert_close(chunked.latent_states,baseline.latent_states,atol=1e-5,rtol=1e-5)
                module.sparse_coder.dictionary[0,0].add_(.2)
                changed=encode(model,inputs)
                torch.testing.assert_close(changed.history_states,baseline.history_states,rtol=0,atol=0)
                torch.testing.assert_close(changed.latent_states[:,1:],baseline.latent_states[:,1:],atol=1e-5,rtol=1e-5)
                self.assertGreater((changed.latent_states[:,0]-baseline.latent_states[:,0]).abs().sum().item(),0)

    def test_summary_loss_trains_dictionary_and_first_block_without_sparse_loss(self):
        for name in ('SASRec','HSTU'):
            model=self.model(name)
            encoded=encode(model,fixture())
            (encoded.latent_states*torch.randn_like(encoded.latent_states)).sum().backward()
            module=model._ndp_module
            self.assertGreater(module.sparse_coder.dictionary.grad.abs().sum().item(),0)
            self.assertTrue(all(p.weight.grad is not None for p in module.token_projections))
            first=(model.attention_layers[0].in_proj_weight if name=='SASRec'
                   else model._hstu._attention_layers[0]._uvqk)
            self.assertGreater(first.grad.abs().sum().item(),0)
            self.assertIsNone(module.raw_projection.weight.grad)

    def test_future_values_and_timestamps_do_not_enter_prefix(self):
        with torch.inference_mode():
            for name in ('SASRec','HSTU'):
                model=self.model(name)
                inputs=fixture()
                original=encode(model,inputs)
                lengths,ids,payloads=inputs
                altered_ids=ids.clone()
                altered_payloads={key:value.clone() for key,value in payloads.items()}
                for row,length in enumerate(lengths):
                    altered_ids[row,length:]=20
                    altered_payloads['timestamps'][row,length:]=999999
                changed=encode(model,(lengths,altered_ids,altered_payloads))
                torch.testing.assert_close(original.history_states,changed.history_states)
                torch.testing.assert_close(original.latent_states,changed.latent_states)
                self.assertTrue(torch.equal(original.selected_positions,changed.selected_positions))

    def test_joint_attention_matches_direct_unpooled_reference(self):
        with torch.inference_mode():
            for candidate_batch in (1,2):
                model=self.model()
                module=model._ndp_module
                inputs=fixture()
                encoded=encode(model,inputs)
                memory=module.finish_user_memory(encoded)
                items=module.prepare_items(torch.randn(candidate_batch,3,6))
                actual,weights,joint=module.attend_prepared(memory,items,return_weights=True)
                for row in range(2):
                    for candidate in range(3):
                        item_row=0 if candidate_batch==1 else row
                        length=int(inputs[0][row])
                        recent=list(range(max(0,length-3),length))
                        union=set()
                        summary=[]
                        for expert in range(2):
                            for k,atom in enumerate(items.active_indices[item_row,candidate,expert]):
                                if items.active_mask[item_row,candidate,expert,k]:
                                    slot=expert*5+int(atom)
                                    union.update(encoded.selected_positions[row,slot][encoded.selected_mask[row,slot]].tolist())
                                    summary.append(module.summary_projections[expert](encoded.latent_states[row,slot]))
                        positions=recent+sorted(union) # Preserve recent/selected multiplicity.
                        original=module.raw_projection(encoded.original[row,positions])
                        tokens=torch.cat([original,torch.stack(summary)],0)
                        keys,values=module.memory_key(tokens),module.memory_value(tokens)
                        query=module.query_projection(items.queries[item_row,candidate])
                        probability=(query@keys.T/module.attention_dim**.5).softmax(-1)
                        expected=probability@values
                        torch.testing.assert_close(actual[row,candidate],expected,atol=2e-6,rtol=2e-5)
                        self.assertEqual(int(joint.mask[row,candidate].sum()),len(positions)+len(summary))
                torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights.sum(-1)))

    def test_pi_only_mixes_final_scores_and_alpha_is_diagnostic(self):
        model=self.model()
        module=model._ndp_module
        inputs=fixture()
        memory=module.prepare_user(model,*inputs[:2],model.get_item_embeddings(inputs[1]),inputs[2])
        items=module.prepare_items(torch.randn(1,2,6))
        outputs,weights,_=module.attend_prepared(memory,items,True)
        logits=module.expert_logits(outputs,items)
        different=replace(items,expert_weights=items.expert_weights.flip(-1),
                          activation_weights=torch.rand_like(items.activation_weights))
        other,other_weights,_=module.attend_prepared(memory,different,True)
        torch.testing.assert_close(outputs,other)
        torch.testing.assert_close(weights,other_weights)
        scores=module.score_prepared(memory,different)[0]
        torch.testing.assert_close(scores,(logits*different.expert_weights).sum(-1))

    def test_empty_short_histories_and_memory_segment_ablations(self):
        for name in ('SASRec','HSTU'):
            for length in (0,1,3):
                for settings in ({},{'recent_events':0},{'use_selected_tokens':False,'use_summary_tokens':False}):
                    model=self.model(name,**settings)
                    ids=torch.zeros(1,11,dtype=torch.long)
                    ids[:,:length]=torch.arange(1,length+1)
                    module=model._ndp_module
                    _,memory=model.encode_with_history(torch.tensor([length]),ids,model.get_item_embeddings(ids),{})
                    items=module.prepare_items(torch.randn(1,1,6))
                    output,weights,_=module.attend_prepared(memory,items,True)
                    score=module.score_prepared(memory,items)[0]
                    self.assertTrue(torch.isfinite(score).all())
                    if length==0:
                        self.assertEqual(output.abs().sum().item(),0)
                        self.assertEqual(weights.abs().sum().item(),0)

    def test_cached_history_reuses_native_values_across_dictionary_updates(self):
        for name in ('SASRec','HSTU'):
            with torch.inference_mode():
                model=self.model(name)
                inputs=fixture()
                first=encode(model,inputs)
                self.assertEqual(len(first.history_cache.layers),model._num_blocks)
                model._ndp_module.sparse_coder.dictionary.add_(.05)
                second=encode(model,inputs,history_cache=first.history_cache)
                fresh=encode(model,inputs)
                torch.testing.assert_close(second.latent_states,fresh.latent_states)
                self.assertEqual(first.history_states.data_ptr(),second.history_states.data_ptr())
                torch.testing.assert_close(first.history_states,second.history_states)
                for a,b in zip(first.history_cache.layers,second.history_cache.layers):
                    self.assertEqual(a['k'].data_ptr(),b['k'].data_ptr())
                self.assertGreater((first.latent_states-second.latent_states).abs().sum().item(),0)
                inputs[2]['request_timestamps'].add_(100)
                changed_time=encode(model,inputs,history_cache=first.history_cache)
                direct_time=encode(model,inputs)
                torch.testing.assert_close(changed_time.latent_states,direct_time.latent_states)
                self.assertNotEqual(first.history_states.data_ptr(),changed_time.history_states.data_ptr())
                with torch.no_grad():
                    next(model._embedding_module.parameters()).add_(.1)
                with self.assertRaisesRegex(ValueError,'Stale or foreign'):
                    encode(model,inputs,history_cache=first.history_cache)

    def test_slicing_serialization_staleness_and_exact_retrieval(self):
        for name in ('SASRec','HSTU'):
            model=self.model(name)
            inputs=fixture()
            module=model._ndp_module
            with torch.inference_mode():
                h,memory=model.encode_with_history(inputs[0],inputs[1],model.get_item_embeddings(inputs[1]),inputs[2])
                embeddings=torch.randn(1,9,6)
                ids=torch.arange(101,110)[None]
                items=module.prepare_items(embeddings,ids)
                score=module.score_prepared(memory,items)[0]
                parts=torch.cat([module.score_prepared(memory,items.slice_candidates(i,i+2))[0]
                                 for i in range(0,9,2)],1)
                torch.testing.assert_close(parts,score)
                one=module.score_prepared(memory.index_batch(-1),items)[0]
                torch.testing.assert_close(one,score[-1:])
                top=LearnedBruteForceTopK(module,embeddings,ids,candidate_chunk_size=2)
                values,result=top(h,3,user_memory=memory)
                expected,indices=score.topk(3,1)
                torch.testing.assert_close(values,expected)
                self.assertTrue(torch.equal(result,ids.expand(2,-1).gather(1,indices)))
                saved=io.BytesIO()
                torch.save(model.state_dict(),saved)
                saved.seek(0)
                model.load_state_dict(torch.load(saved,weights_only=True),strict=True)
                with self.assertRaisesRegex(ValueError,'Stale'):
                    module.score_prepared(memory,items)
                h,memory=model.encode_with_history(inputs[0],inputs[1],model.get_item_embeddings(inputs[1]),inputs[2])
                torch.testing.assert_close(module.score_prepared(memory,module.prepare_items(embeddings,ids))[0],score)
                bad=module.get_extra_state(); bad['format_version']=2
                with self.assertRaisesRegex(ValueError,'version 3'):
                    module.set_extra_state(bad)

    def test_chunking_preserves_recommendation_and_sparse_gradients(self):
        model=self.model().train()
        inputs=fixture()
        embeddings=torch.randn(2,5,6)
        def backward(chunk):
            torch.manual_seed(333)
            model.zero_grad(set_to_none=True)
            _,memory=model.encode_with_history(inputs[0],inputs[1],model.get_item_embeddings(inputs[1]),inputs[2])
            module=model._ndp_module
            items=module.prepare_items(embeddings)
            scores=torch.cat([module.score_prepared(memory,items.slice_candidates(i,i+chunk))[0]
                              for i in range(0,5,chunk)],1)
            (scores.square().mean()+.1*items.auxiliary_losses()['sparse_code_loss']).backward()
            return scores.detach(),{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        first,first_grad=backward(5)
        other,other_grad=backward(2)
        torch.testing.assert_close(first,other)
        self.assertEqual(first_grad.keys(),other_grad.keys())
        for name in first_grad:
            torch.testing.assert_close(first_grad[name],other_grad[name],atol=3e-6,rtol=3e-4,msg=name)

    def test_third_backbone_can_register_without_inheriting_repository_bridge(self):
        class ToyHost(nn.Module):
            def __init__(self):
                super().__init__(); self.linear=nn.Linear(6,6)
        class ToyAdapter:
            num_layers=1; activation_checkpoint=False
            def __init__(self,host): self.host=host
            def preprocess(self,lengths,ids,embeddings,payloads):
                mask=torch.arange(ids.shape[1])[None]<lengths[:,None]
                return embeddings*mask[...,None],mask
            def target_positions(self,lengths): return torch.zeros(lengths.shape[0],6)
            def postprocess(self,states): return states
            def history_step(self,index,history,mask,lengths,payloads):
                return self.host.linear(history),dict(k=history,v=history,mask=mask)
            def latent_step(self,index,z,context):
                logits=z@context['k'].transpose(-1,-2)
                weights=torch.cat([logits,(z*z).sum(-1,keepdim=True)],-1).softmax(-1)
                result=weights[...,:-1]@context['v']+weights[...,-1:]*z
                return self.host.linear(z+result),weights[...,:-1].unsqueeze(1)
        register_backbone_adapter(ToyHost,ToyAdapter)
        host=ToyHost().eval()
        module=lcar().eval()
        inputs=fixture()
        memory=module.prepare_user(host,inputs[0],inputs[1],torch.randn(2,11,6),inputs[2])
        items=module.prepare_items(torch.randn(1,2,6))
        score=module.score_prepared(memory,items)[0]
        self.assertEqual(score.shape,(2,2))
        self.assertFalse(any('host' in key or 'backbone' in key for key in module.state_dict()))

    def test_standalone_lcar_on_unmodified_dot_product_hosts(self):
        from rails.similarities.dot_product_similarity_fn import DotProductSimilarity
        for name in ('SASRec','HSTU'):
            host=small_backbone(name,interaction=DotProductSimilarity()).eval()
            addon=lcar().eval()
            inputs=fixture()
            original_keys=set(host.state_dict())
            memory=addon.prepare_user(host,inputs[0],inputs[1],host.get_item_embeddings(inputs[1]),inputs[2])
            items=addon.prepare_items(torch.randn(1,2,6))
            addon.score_prepared(memory,items)[0].sum().backward()
            self.assertIsInstance(host._ndp_module,DotProductSimilarity)
            self.assertEqual(set(host.state_dict()),original_keys)
            self.assertGreater(addon.sparse_coder.dictionary.grad.abs().sum().item(),0)
            self.assertTrue(any(p.grad is not None for p in host.parameters()))

    def test_lcar_configs_and_public_registration(self):
        import gin
        import train
        from modeling.similarity_utils import get_similarity_function
        from playground.model_registry import MODEL_SPECS
        for name in ('sasrec','hstu'):
            gin.clear_config()
            gin.parse_config_file(str(ROOT/f'configs/ml-1m/{name}-lcar.gin'))
            scorer,description=get_similarity_function('LCAR',50,50)
            self.assertEqual(scorer.scorer_type,'LCAR')
            self.assertEqual((scorer.dictionary_size,scorer.max_active_latents,scorer.recent_events),(512,16,32))
            self.assertEqual(scorer.expert_scorers[0][0].in_features,96)
            self.assertIn('LCAR-v3',description)
            self.assertEqual(MODEL_SPECS[name+'_lcar'].scorer,'LCAR')
            self.assertEqual(gin.query_parameter('train_fn.supervision_mode'),'sampled_prefix')
        gin.clear_config()

    def test_checkpointing_and_sampled_prefix_ddp(self):
        import tempfile
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from modeling.sequential.autoregressive_losses import LocalNegativesSampler
        from modeling.sequential.losses.sampled_softmax import SampledSoftmaxLoss
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo',init_method='file://'+tmp+'/group',rank=0,world_size=1)
            try:
                for name in ('SASRec','HSTU'):
                    model=small_backbone(name,interaction=lcar(),activation_checkpoint=True).train()
                    ddp=DistributedDataParallel(model,find_unused_parameters=True)
                    sampler=LocalNegativesSampler(40,model._embedding_module._item_emb,list(range(1,41)),True,1e-6)
                    loss_fn=SampledSoftmaxLoss(3,1.,model,activation_checkpoint=True,candidate_chunk_size=2)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                    inputs=fixture()
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        loss,aux=ddp(inputs[0],inputs[1],model.get_item_embeddings(inputs[1]),inputs[2],
                                     sampled_targets=torch.tensor([9,10]),negatives_sampler=sampler,sampled_loss=loss_fn)
                        total=loss+.1*aux['sparse_code_loss']
                        total.backward()
                        self.assertTrue(torch.isfinite(total))
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                        self.assertGreater(model._ndp_module.sparse_coder.dictionary.grad.abs().sum().item(),0)
                        optimizer.step()
                        model._ndp_module.post_optimizer_step()
            finally:
                dist.destroy_process_group()

    @unittest.skipUnless(torch.cuda.is_available(),'Configured real-data integration uses CUDA')
    def test_both_configured_backbones_real_prefix_update_reload_and_microbatch_eval(self):
        import gin
        from data.reco_dataset import get_reco_dataset
        from data.eval import get_eval_state,eval_metrics_v2_from_tensors
        from indexing.utils_rails import get_top_k_module
        from modeling.sequential.features import movielens_seq_features_from_row
        from modeling.sequential.prefix_sampling import sample_causal_prefix
        from modeling.sequential.autoregressive_losses import LocalNegativesSampler
        from modeling.sequential.losses.sampled_softmax import SampledSoftmaxLoss
        from torch.utils.data._utils.collate import default_collate
        if not (ROOT/'tmp/ml-1m/sasrec_format.csv').exists(): self.skipTest('Local fixture absent; no downloads')
        with torch.compiler.set_stance('force_eager'):
            dataset=get_reco_dataset('ml-1m',200,chronological=True)
            row=default_collate([dataset.train_dataset[i] for i in (0,1)])
            features,targets,_=movielens_seq_features_from_row(row,'cuda',11)
            for name in ('sasrec','hstu'):
                model=configured_backbone(ROOT/f'configs/ml-1m/{name}-lcar.gin',dataset.max_item_id,'cuda')
                prefix,next_ids=sample_causal_prefix(features,targets,torch.Generator(device='cuda').manual_seed(7))
                sampler=LocalNegativesSampler(dataset.max_item_id,model._embedding_module._item_emb,
                                               dataset.all_item_ids,True,1e-6).cuda()
                criterion=SampledSoftmaxLoss(3,1.,model,activation_checkpoint=True,candidate_chunk_size=2)
                model.train()
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                loss,aux=model(prefix.past_lengths,prefix.past_ids,model.get_item_embeddings(prefix.past_ids),
                               prefix.past_payloads,sampled_targets=next_ids,negatives_sampler=sampler,sampled_loss=criterion)
                total=loss+.1*aux['sparse_code_loss']; total.backward()
                self.assertTrue(torch.isfinite(total))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step(); model._ndp_module.post_optimizer_step(); model.eval()
                with torch.inference_mode():
                    _,memory=model.encode_with_history(prefix.past_lengths,prefix.past_ids,
                                                        model.get_item_embeddings(prefix.past_ids),prefix.past_payloads)
                    self.assertEqual(memory.summaries.shape[1:],(2,512,32))
                    checkpoint=io.BytesIO(); torch.save(model.state_dict(),checkpoint); checkpoint.seek(0)
                    model.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True)
                    state=get_eval_state(model,dataset.all_item_ids[:12],sampler,
                        lambda emb,ids:get_top_k_module('LearnedBruteForceTopK',model,emb,ids,candidate_chunk_size=3),'cuda')
                    target=torch.tensor(dataset.all_item_ids[:1],device='cuda')[None].expand(2,-1)
                    first=eval_metrics_v2_from_tensors(state,model,prefix,target,user_max_batch_size=1)
                    second=eval_metrics_v2_from_tensors(state,model,prefix,target,user_max_batch_size=2)
                    for key in first: torch.testing.assert_close(first[key],second[key])
                gin.clear_config()

    def test_prepared_accounting_reconciles_without_benchmark_execution(self):
        from playground.benchmark_inference_flops import LCARInference,counted
        from playground.inference_flops_counter import sum_totals
        for name in ('SASRec','HSTU'):
            model=self.model(name)
            inputs=fixture()
            raw_fixture=(inputs[1],inputs[0],inputs[2])
            adapter=LCARInference(model,candidate_chunk_size=2)
            try:
                with torch.inference_mode(),torch.compiler.set_stance('force_eager'):
                    context,_,_=counted(adapter,adapter.prepare_model,'model_only')
                    self.assertGreater(context.num_bytes(),0)
                    user,u,_=counted(adapter,lambda:adapter.prepare_user(raw_fixture),'U')
                    items,i,_=counted(adapter,lambda:adapter.prepare_items(torch.arange(1,4)),'I')
                    actual,s,calls=counted(adapter,lambda:adapter.score(user,items),'S')
                    self.assertTrue(all(value==0 for value in calls.values()))
                    composed,total,_=counted(adapter,lambda:adapter.score(adapter.prepare_user(raw_fixture),
                                                           adapter.prepare_items(torch.arange(1,4))),'composed')
                    torch.testing.assert_close(composed,actual)
                    self.assertEqual(total.totals(),sum_totals(u.totals(),i.totals(),s.totals()))
            finally:
                adapter.close()


if __name__=='__main__':
    unittest.main()
