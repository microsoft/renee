import torch
import numpy as np
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
import math
import random

import transformers
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from scipy.sparse import csr_matrix, save_npz
import apex
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import pandas as pd
import torch.distributed as dist
import datetime
import os.path
#import numba as nb
try:
    import xfc_gemm_cuda
    
except ModuleNotFoundError:
    # Error handling
    # needed only for custom-cuda
    pass


#from plasma_utils import *
def noop(func):
    return func
profile = noop


def printacc(score_mat, K = 5, X_Y = None, disp = True, inv_prop_ = None):
    if X_Y is None: X_Y = tst_X_Y
    if inv_prop_ is None: inv_prop_ = inv_prop

    acc = xc_metrics.Metrics(X_Y.tocsr().astype(np.bool), inv_prop_)
    metrics = np.array(acc.eval(score_mat, K))*100
    df = pd.DataFrame(metrics)

    if inv_prop_ is None : df.index = ['P', 'nDCG']
    else : df.index = ['P', 'nDCG', 'PSP', 'PSnDCG']

    df.columns = [str(i+1) for i in range(K)]
    if disp: display(df.round(2))
    return df

def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()
    
    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat

def bert_fts_batch_to_tensor(input_ids, attention_mask):
    maxlen = attention_mask.sum(axis=1).max()
#     return {'input_ids': torch.from_numpy(input_ids[:, :maxlen]), 
#             'attention_mask': torch.from_numpy(attention_mask[:, :maxlen])}
    return {'input_ids': torch.from_numpy(input_ids), 
            'attention_mask': torch.from_numpy(attention_mask)}
    #return {'input_ids': torch.from_numpy(input_ids), 
    #        'attention_mask': torch.from_numpy(attention_mask)}

def csr_to_pad_tensor(spmat, pad):
    maxlen = spmat.getnnz(1).max()
    ret = {'inds': torch.full((spmat.shape[0], maxlen), pad).long().flatten(),
           'vals': torch.zeros(spmat.shape[0], maxlen).flatten()}
    ptrs = []
    for i in range(spmat.shape[0]):
        ptrs.append(torch.arange(i*maxlen, i*maxlen + spmat.indptr[i+1] - spmat.indptr[i]))
    ptrs = torch.cat(ptrs)
    ret['inds'][ptrs] = torch.LongTensor(spmat.indices)
    ret['inds'] = ret['inds'].reshape((spmat.shape[0], maxlen))
    ret['vals'][ptrs] = torch.Tensor(spmat.data)
    ret['vals'] = ret['vals'].reshape((spmat.shape[0], maxlen))
    return ret

class FP32Linear(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(FP32Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.transformer_weight = Parameter(torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.transformer_bias = Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('transformer_bias', None)
        self.reset_parameters()

    @profile
    def forward(self, input):
        #print("before linear",torch.cuda.memory_summary())
        #torch.cuda.reset_max_memory_allocated()
        #sys.stdout.flush()
        with torch.cuda.amp.autocast(enabled=False):
            out = F.linear(input.float(), self.transformer_weight, self.transformer_bias) 
        return out
        

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.transformer_weight.size(1))
        self.transformer_weight.data.uniform_(-stdv, stdv)
        if self.transformer_bias is not None:
            self.transformer_bias.data.uniform_(-stdv, stdv)


class TransformerInputLayer(nn.Module):
    def __init__(self, transformer):
        super(TransformerInputLayer, self).__init__()
        self.transformer = transformer
        self.pooler = self.create_pooler()
      
    @profile
    def forward(self, data):
        return self.pooler(self.transformer(**data),data).contiguous()
    
    def create_pooler(self):
        def f(tf_output, batch_data):
            last_hidden_state = tf_output['last_hidden_state']
            input_mask_expanded = batch_data['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            return sum_hidden_state / sum_mask
        return f
      

LOSS_SAMPLE_FREQ = 100 

class GenericModel(nn.Sequential):
    def __init__(self, rank, args, numy, numy_per_gpu, per_gpu_batch_size, modules: Iterable[nn.Module] = None,  device: str = None, name: str = 'generic_model', out_dir: str = None):
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
        super().__init__()
        self.embed = nn.Sequential(modules)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rank = rank
        self.world_size = args.world_size
        self.scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32encoder, init_scale =2**12 if args.fp16xfc else 2**16) 
        self.numy = numy
        self.numy_per_gpu = numy_per_gpu   
        self.padded_numy =  numy_per_gpu*args.world_size   
        self.exp_batch_size = per_gpu_batch_size  # default expected batch size, current batch size may be different due to final, partial batch
        self.xfc_batch_size = per_gpu_batch_size*args.world_size//args.accum
        self.fp16encoder = not args.fp32encoder
        self.fp16xfc = args.fp16xfc
        self.accum = args.accum 
        self.compute_loss = not args.noloss
        self.checkpoint_resume = args.checkpoint_resume
        self.custom_cuda = args.custom_cuda
        self.default_impl = args.default_impl
        self.default_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.count = 0
        # early explicit allocation for large variables to avoid OOM due to mem allocator fragmentation
        # doesn't work with autocast 
        if not self.default_impl:
          self.outsoft = torch.empty((self.xfc_batch_size, self.numy_per_gpu), dtype = torch.float16 if args.fp16xfc else torch.float32, device=device)
          self.grad_input = torch.empty((per_gpu_batch_size*args.world_size, args.bottleneck_dims), dtype=torch.float32,device=device)
          # torch.addmm requires this param with beta=0!
          self.dummy = torch.zeros(1, dtype = torch.float16 if args.fp16xfc else torch.float32, device=device)
        if self.world_size > 1:
            self.gather_list = [torch.empty(self.exp_batch_size, args.bottleneck_dims, dtype=torch.float32, device=device)
                     for _ in range(self.world_size)]

        # xfc layer
        self.xfc_weight = nn.Parameter(torch.Tensor(numy_per_gpu,args.bottleneck_dims))
        #nn.init.kaiming_uniform_(self.xfc_weight, a=math.sqrt(5))
        nn.init.normal_(self.xfc_weight,mean=0.0,std=0.02) # gpt-1 paper says this is fine since layernorm is used throughout embedding
        
        self._target_device = torch.device(device)
        self.name = name
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        

    @profile
    def fit(self,
            dataloader,
            loss_model,
            xfc_optimizer_class: Type[Optimizer],
            xfc_optimizer_params : Dict[str, object],
            tf_optimizer_class: Type[Optimizer],
            tf_optimizer_params : Dict[str, object],
            epochs: int = 1,
            scheduler: str = 'warmupcosine',
            warmup_steps: int = 10000,
            evaluator = None,
            evaluation_epochs: int = 5,
            max_grad_norm: float = -1,
            ):


        loss_model.to(self._target_device)

        self.count = 0
        steps_per_epoch = len(dataloader)
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizer_params_xfc = []
        optimizer_params_tf = []
            
        for n, p in loss_model.named_parameters():
            if p.requires_grad:
                if 'xfc' in n: 
                    optimizer_params_xfc.append((n, p))
                else:
                    optimizer_params_tf.append((n, p))
            
        no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        tf_optimizer_grouped_parameters = [
                {'params': [p for n, p in optimizer_params_tf if not any(nd in n for nd in no_decay_params)], 'weight_decay': tf_optimizer_params['weight_decay']},
                {'params': [p for n, p in optimizer_params_tf if any(nd in n for nd in no_decay_params)], 'weight_decay': 0.0}
        ]
        xfc_optimizer_grouped_parameters = [
                {'params': [p for n, p in optimizer_params_xfc], 'weight_decay': xfc_optimizer_params['weight_decay']},
        ]

        self.xfc_optimizer = xfc_optimizer_class(xfc_optimizer_grouped_parameters, **xfc_optimizer_params)
        self.tf_optimizer = tf_optimizer_class(tf_optimizer_grouped_parameters, **tf_optimizer_params)
        
        self.xfc_scheduler = self._get_scheduler(self.xfc_optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
        self.tf_scheduler = self._get_scheduler(self.tf_optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
     

        data_iterator = iter(dataloader)
        self.epoch = 0
        if self.checkpoint_resume != '':
          if os.path.isfile(f'{self.checkpoint_resume}/{self.name}_{self.rank}_checkpoint.pt'):
            prev_loss = self.resume()
            print('Resuming training from epoch: ', self.epoch, 'with loss: ', prev_loss, flush=True)
          else:
            print('No checkpoint file to resume. Starting training from scratch', flush=True)

        start_epoch =  self.epoch
        loss = torch.tensor(0.0,device=self._target_device)
        total_loss = torch.tensor(0.0,device=self._target_device)
        for epoch in trange(start_epoch, epochs, desc="Epoch", initial=start_epoch, total=epochs):
            self.epoch = epoch
            training_steps = 0
            total_loss = 0.0
            loss_model.zero_grad()
            loss_model.train()

            grad_accum = 0
            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=(self.rank != 0)):
                try:
                    batch_data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    batch_data = next(data_iterator)
                batch_data = self.batch_to_device(batch_data, self._target_device)

                # Unoptimized Implementation -- easier to modify for experimentation since autograd handles backward but perf is 10X worse than optimized version
                if self.default_impl: 
                    #with torch.cuda.amp.autocast(enabled=self.fp16encoder):
                    grad_accum += 1
                    embed = loss_model(batch_data) 
                    embed_out = loss_model.gather_embed(embed, batch_data)
                    bsz = batch_data['batch_size']
                    out = loss_model.xfc_forward(embed_out)
                    batch_data['yfull'] = torch.zeros(batch_data['batch_size'], self.numy_per_gpu+1, device=batch_data['y']['inds'].device).scatter_(1, batch_data['y']['inds'], batch_data['y']['vals'])[:, :-1]
                    loss = self.default_loss(out, batch_data['yfull'])
                    loss.backward()
                    total_loss += loss/(bsz*self.padded_numy)                       
#                     if grad_accum % 1 == 0:
                    if grad_accum % self.accum == 0:
                     grad_accum = 0
                     self.xfc_optimizer.step()
                     self.xfc_optimizer.zero_grad()
                     self.tf_optimizer.step()
                     self.tf_optimizer.zero_grad()
                     self.xfc_scheduler.step()
                     self.tf_scheduler.step()
                     training_steps += 1
                    continue

                # Optimized Implementation 
                # get embedding
                with torch.cuda.amp.autocast(enabled=self.fp16encoder):
                    embed = loss_model(batch_data) 

                # xfc layer custom forward and backward performed with no_grad
                with torch.no_grad():
                    embed_out = loss_model.gather_embed(embed, batch_data)
                    

                    bsz = batch_data['batch_size']
                    pos_x_y = batch_data['z']
                    loss = 0.0
                    if self.accum == 1:
                        do_forward = not self.custom_cuda or (bsz % 8 != 0) or ((self.count % LOSS_SAMPLE_FREQ == 0) and self.compute_loss)
                        # Do xfc forward-backward 
                        #if (not (self.custom_cuda and bsz%8 == 0)):
                        if do_forward:
                          loss_model.xfc_forward(embed_out, self.outsoft[0:bsz,:])
                        loss += loss_model.xfc_backward(embed_out, self.outsoft[0:bsz,:], pos_x_y[0], pos_x_y[1], self.grad_input[0:bsz,:], not do_forward) 
                    else:
                        # Do xfc forward-backward xfcz at a time, accummulating gradients
                        start_xfcz = 0
                        end_xfcz = self.xfc_batch_size
                        if end_xfcz > bsz: # partial batch
                          end_xfcz = bsz
                        index = batch_data['i']
                        i = 0
                        while (end_xfcz <= bsz):
                          #print(bsz, i, start_xfcz, end_xfcz, index[i], index[i+1], flush=True)
                          embed_out_xfcz = embed_out[start_xfcz:end_xfcz,:]
                          do_forward = not self.custom_cuda or ((end_xfcz-start_xfcz)%8 != 0) or ((self.count % LOSS_SAMPLE_FREQ == 0) and self.compute_loss)
                          #if (not (self.custom_cuda and (end_xfcz-start_xfcz)%8 == 0)):
                          if do_forward:
                            loss_model.xfc_forward(embed_out_xfcz, self.outsoft[0:end_xfcz-start_xfcz,:])
                          loss += loss_model.xfc_backward(embed_out_xfcz, self.outsoft[0:end_xfcz-start_xfcz,:],
                                                     pos_x_y[0,index[i]:index[i+1]],
                                                     pos_x_y[1,index[i]:index[i+1]],
                                                     self.grad_input[start_xfcz:end_xfcz,:], not do_forward) #(self.custom_cuda and (end_xfcz-start_xfcz)%8 == 0))
                          i += 1
                          start_xfcz = end_xfcz
                          end_xfcz += self.xfc_batch_size
                          if start_xfcz < bsz and end_xfcz > bsz:
                            end_xfcz = bsz

                    # Update xfc layer (without scaling), free up gradient
                    self.xfc_optimizer.step()
                    self.xfc_weight.grad = None
                   

                # now do the backward for the embed layers
                per_gpu_bsz = bsz//self.world_size
                remainder = bsz % self.world_size
                startbs = self.rank * per_gpu_bsz
                endbs = startbs + per_gpu_bsz
                if remainder > 0:
                  if (self.rank < remainder): # add remainder to first n ranks
                    startbs += self.rank
                    endbs += self.rank + 1
                  else:
                    startbs += remainder
                    endbs += remainder

                embed.backward(self.grad_input[startbs:endbs,:])
                if self.compute_loss and self.count % LOSS_SAMPLE_FREQ == 0:
                  loss /= (bsz*self.padded_numy)
                  total_loss += loss                       
                  training_steps += 1
                self.count += 1

                if max_grad_norm > 0: 
                    self.scaler.unscale_(self.tf_optimizer)
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                # finish update with tf_optimizer
                self.scaler.step(self.tf_optimizer)
                self.tf_optimizer.zero_grad()
                self.scaler.update()

                self.xfc_scheduler.step()
                self.tf_scheduler.step()

                #training_steps += 1
                del batch_data
            
            if self.compute_loss:
              mean_loss = total_loss.item()/training_steps
            else:
              mean_loss = 0.0
            if self.rank == 0:
                print(f'mean loss after epoch {epoch} : {"%.4E"%(mean_loss)}')
                print("Scale: ",self.scaler.get_scale())
                sys.stdout.flush()
            if (((epoch + 1) % evaluation_epochs == 0) or (epoch == epochs - 1)) and evaluator is not None:
                score = evaluator(loss_model, epoch, mean_loss, self.out_dir, self.name)
            if self.checkpoint_resume != '':
                self.checkpoint(mean_loss)
                    
    @staticmethod
    def _get_scheduler(optimizer, warmup_steps: int, t_total: int):
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
            
    def batch_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        if isinstance(batch, Dict):
            for outkey in batch:
                if isinstance(batch[outkey], torch.Tensor):
                    batch[outkey] = batch[outkey].to(device)
                if isinstance(batch[outkey], Dict):
                    for inkey in batch[outkey]:
                        if isinstance(batch[outkey][inkey], torch.Tensor):
                            batch[outkey][inkey] = batch[outkey][inkey].to(device)
        return batch

    def checkpoint(self, loss):
        out_dir = self.checkpoint_resume
        os.makedirs(out_dir, exist_ok=True)
        save_dict = {'epoch': self.epoch, 'model_dict': self.state_dict(), 'loss': loss, 
                     'scaler': self.scaler.state_dict(),
                     'py_rng': random.getstate(),
                     'torch_rng' : torch.get_rng_state(),
                     'torch_cuda_rng' : torch.cuda.get_rng_state(),
                     'torch_random_rng' : torch.random.get_rng_state(),
                     'np_rng' : np.random.get_state()}
        save_dict['xfc_optimizer_state_dict'] = self.xfc_optimizer.state_dict()
        save_dict['tf_optimizer_state_dict']  = self.tf_optimizer.state_dict()
        save_dict['xfc_scheduler_state_dict'] = self.xfc_scheduler.state_dict()
        save_dict['tf_scheduler_state_dict']  = self.tf_scheduler.state_dict()
        if os.path.exists(f'{out_dir}/{self.name}_{self.rank}_checkpoint.pt'):
            os.rename(f'{out_dir}/{self.name}_{self.rank}_checkpoint.pt',f'{out_dir}/{self.name}_{self.rank}_checkpoint_prev.pt')
        torch.save(save_dict, f'{out_dir}/{self.name}_{self.rank}_checkpoint.pt')

    def resume(self):
        out_dir = self.checkpoint_resume
        load_dict = torch.load(f'{out_dir}/{self.name}_{self.rank}_checkpoint.pt')
        self.epoch = load_dict['epoch'] + 1
        self.load_state_dict(load_dict['model_dict'])
        loss = load_dict['loss']
        self.scaler.load_state_dict(load_dict['scaler'])
        random.setstate(load_dict['py_rng'])
        torch.set_rng_state(load_dict['torch_rng'])
        torch.cuda.set_rng_state(load_dict['torch_cuda_rng'])
        torch.random.set_rng_state(load_dict['torch_random_rng'])
        np.random.set_state(load_dict['np_rng'])
        self.xfc_optimizer.load_state_dict(load_dict['xfc_optimizer_state_dict'])
        self.tf_optimizer.load_state_dict(load_dict['tf_optimizer_state_dict'])
        self.xfc_scheduler.load_state_dict(load_dict['xfc_scheduler_state_dict'])
        self.tf_scheduler.load_state_dict(load_dict['tf_scheduler_state_dict'])
        return loss
    
    def save(self, out_dir: str = None):
        if out_dir is None: out_dir = self.out_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), f'{out_dir}/{self.name}_{self.rank}_state_dict.pt')
    
    def load(self, out_dir: str = None):
        if out_dir is None: out_dir = self.out_dir
        self.load_state_dict(torch.load(f'{out_dir}/{self.name}_{self.rank}_state_dict.pt', map_location=self._target_device))

class AllGather(torch.autograd.Function):
    """ 
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank] 


all_gather = AllGather.apply

# Default Multi node case
class BCELossMultiNodeDefault(nn.Module):
    def __init__(self, model: GenericModel):
        super(BCELossMultiNodeDefault, self).__init__()
        self.model = model
 
    def forward(self, batch_data):
        embed = self.model.embed(batch_data['xfts'])
        return embed

    def gather_embed(self, embed, batch_data):
      if self.model.world_size > 1:
        # all-gather embeddings to model-parallel gpus
        self.global_batch_size = batch_data['batch_size']
        self.remainder = self.global_batch_size % self.model.world_size
        self.per_gpu_batch_size = self.global_batch_size//self.model.world_size   # this may differ from self.exp_batch_size for final batch, with or without remainders

        my_cur_batch_size = embed.size()[0]   # this may differ from per_gpu_batch_size only in final partial batch, due to remainder in last rank
     
        if (self.per_gpu_batch_size != self.model.exp_batch_size): # Final, partial batch of epoch
            # Pad partial batch to expected batch size by repating cur_batch_size for all_gather to work
            embed = embed.repeat((self.model.exp_batch_size + my_cur_batch_size - 1)//my_cur_batch_size,1)[0:self.model.exp_batch_size,:]
            all_gather(self.model.gather_list, embed)
            # Extract out correct partial batches from padded batches
            embed_out = torch.cat([self.model.gather_list[i][0:j,:] 
                                    for i,j in [ (i, self.per_gpu_batch_size + 1) 
                                        if i < self.remainder else (i, self.per_gpu_batch_size) 
                                        for i in range(self.model.world_size)]
                                    ], dim=0)
        else:
            all_gather(self.model.gather_list, embed) 
            embed_out = torch.cat(self.model.gather_list, dim=0)
        return embed_out
      else:
        return embed

    def xfc_forward(self, embed):
        # run fully-connected layer in model-parallel manner
        out = torch.matmul(embed, self.model.xfc_weight.t())
        return out

# Multi node case
class BCELossMultiNode(nn.Module):
    def __init__(self, model: GenericModel):
        super(BCELossMultiNode, self).__init__()
        self.model = model
        self.loss = torch.tensor(0.0,device=model._target_device)
 
    def forward(self, batch_data):
        embed = self.model.embed(batch_data['xfts'])
        return embed

    def gather_embed(self, embed, batch_data):
      if self.model.world_size > 1:
        # all-gather embeddings to model-parallel gpus
        self.global_batch_size = batch_data['batch_size']
        self.remainder = self.global_batch_size % self.model.world_size
        self.per_gpu_batch_size = self.global_batch_size//self.model.world_size   # this may differ from self.exp_batch_size for final batch, with or without remainders

        my_cur_batch_size = embed.size()[0]   # this may differ from per_gpu_batch_size only in final partial batch, due to remainder in last rank
     
        if (self.per_gpu_batch_size != self.model.exp_batch_size): # Final, partial batch of epoch
            # Pad partial batch to expected batch size by repating cur_batch_size for all_gather to work
            embed = embed.repeat((self.model.exp_batch_size + my_cur_batch_size - 1)//my_cur_batch_size,1)[0:self.model.exp_batch_size,:]
            dist.all_gather(self.model.gather_list, embed)
            # Extract out correct partial batches from padded batches
            embed_out = torch.cat([self.model.gather_list[i][0:j,:] 
                                    for i,j in [ (i, self.per_gpu_batch_size + 1) 
                                        if i < self.remainder else (i, self.per_gpu_batch_size) 
                                        for i in range(self.model.world_size)]
                                    ], dim=0)
        else:
            dist.all_gather(self.model.gather_list, embed) 
            embed_out = torch.cat(self.model.gather_list, dim=0)
      else:
        embed_out = embed

      if self.model.fp16xfc: # do fp16 conversions once
            embed_out = embed_out.to(torch.float16)  
            self.xfc_weight = self.model.xfc_weight.to(torch.float16)
            
      else:
            self.xfc_weight = self.model.xfc_weight  
      return embed_out

    def xfc_forward(self, embed, out):
        # run fully-connected layer in model-parallel manner
        # autocast doesn't work when out tensor is specified!
       
        torch.matmul(embed, self.xfc_weight.t(), out=out)
        return  


    def xfc_backward(self, embed_out, out, pos_x, pos_y, grad_input, use_custom):         
        # Compute loss, do backward pass
        if self.model.compute_loss and self.model.count % LOSS_SAMPLE_FREQ == 0:
          self.loss = out.clamp(min=0.0).sum(dtype=torch.float32) 
          self.loss -= out[pos_x,pos_y].sum(dtype=torch.float32) 
        
          #with torch.cuda.amp.autocast(enabled=self.model.fp16): # TODO: more accurate but needs lots of memory (creates fp32 out?), need a custom cuda kernel!
          self.loss += (1+(-torch.abs(out.float())).exp()).log().sum(dtype=torch.float32)

          # async all_reduce on loss to get global loss across model_parallel workers
          if self.model.world_size > 1:
            loss_work = dist.all_reduce(self.loss, dist.ReduceOp.SUM, async_op=True)

        # loss backward
        if use_custom:
          xfc_gemm_cuda.xfc_gemm(embed_out,self.xfc_weight.t(),out,1.0,0.0,True)
        else:
          torch.sigmoid(out,out=out) # TODO: combine this with forward matmul, will save 10+%
        out[pos_x,pos_y] -= 1.0

        #mean reduction for BCEloss implies gradients are also averaged but this does not work well, going with sum
        #out.mul_(1.0/(self.model.numy_per_gpu*self.global_batch_size)) # mean reduction
        if self.model.xfc_weight.grad is None:
           self.model.scaler.scale(self.loss)
           self.scale_bwd = self.model.scaler.get_scale()
        # manual backward through fully-connected layer
        if self.model.fp16xfc:
          if False and use_custom:
            xfc_gemm_cuda.xfc_gemm(out,self.xfc_weight,grad_input,self.scale_bwd,0.0,False)
          else:
            grad_input.copy_(torch.addmm(self.model.dummy,out,self.xfc_weight,beta=0,alpha=self.scale_bwd).to(torch.float32))
        else:
          torch.addmm(self.model.dummy,out,self.xfc_weight,beta=0,alpha=self.scale_bwd,out=grad_input)

        if self.model.compute_loss and self.model.count % LOSS_SAMPLE_FREQ == 0:
          if self.model.world_size > 1:
            loss_work.wait() # get global loss
        # print(loss)

        if self.model.world_size > 1:
          # async all-reduce grad_input from all GPUs: TODO: reduce_scatter is sufficient
          work = dist.all_reduce(grad_input, dist.ReduceOp.SUM, async_op=True)

        # matmul for computing gradient w.r.t weights
        if self.model.xfc_weight.grad is None:
          if use_custom:
             self.model.xfc_weight.grad = torch.empty_like(self.model.xfc_weight)
             xfc_gemm_cuda.xfc_gemm(out.t(),embed_out,self.model.xfc_weight.grad,1.0,0.0,False)
          else:
            self.model.xfc_weight.grad = out.t().mm(embed_out).to(torch.float32)
        else:
           if self.model.fp16xfc:
             if use_custom:
               xfc_gemm_cuda.xfc_gemm(out.t(),embed_out,self.model.xfc_weight.grad,1.0,1.0,False)
             else:
               self.model.xfc_weight.grad += torch.addmm(self.model.dummy,out.t(),embed_out,beta=0,alpha=1).to(torch.float32)
           else:
             torch.addmm(self.model.xfc_weight.grad,out.t(),embed_out,beta=1,alpha=1,out=self.model.xfc_weight.grad)
           #self.model.xfc_weight.grad += out.t().mm(embed_out).to(torch.float32)

       
        if self.model.world_size > 1:
          # wait for grad_input
          work.wait() 

        # print('loss =',loss.item()) 
        return self.loss

# Single node case
class BCELoss(nn.Module):
    def __init__(self, model: GenericModel, reduction='mean'):
        super(BCELoss, self).__init__()
        self.model = model
        self.loss = torch.tensor(0.0,device=model._target_device)

    def forward(self, batch_data):
        embed = self.model.embed(batch_data['xfts'])
        return embed

    def gather_embed(self, embed, batch_data):
        # in single node case, no need to gather but do fp16 conversions once!
        if self.model.fp16xfc: 
            embed_out = embed.to(torch.float16) 
            self.xfc_weight = self.model.xfc_weight.to(torch.float16)
        else:
            embed_out = embed 
            self.xfc_weight = self.model.xfc_weight
        return embed_out 

    def xfc_forward(self, embed, out):
        torch.matmul(embed, self.xfc_weight.t(), out=out)
        return


    def xfc_backward(self, embed_out, out, pos_x, pos_y, grad_input, use_custom):  #use_custom not supported yet TBD
        if self.model.compute_loss:
          self.loss = out.clamp(min=0.0).sum(dtype=torch.float32) 
          self.loss -= out[pos_x,pos_y].sum(dtype=torch.float32) 
          self.loss += (1+(-torch.abs(out)).exp()).log().sum(dtype=torch.float32)

        if self.model.xfc_weight.grad is None:
          self.model.scaler.scale(self.loss)
          self.scale_bwd = self.model.scaler.get_scale()

        # loss backward
        torch.sigmoid(out, out=out)
        out[pos_x,pos_y] -= 1.0

        # linear backward
        if self.model.fp16xfc:
            grad_input.copy_(torch.addmm(self.model.dummy,out,self.xfc_weight,beta=0,alpha=self.scale_bwd).to(torch.float32))
        else:
            torch.addmm(self.model.dummy,out,self.xfc_weight,beta=0,alpha=self.scale_bwd,out=grad_input)
        if self.model.xfc_weight.grad is None:
           self.model.xfc_weight.grad = out.t().mm(embed_out).to(torch.float32)
        else:
            if self.model.fp16xfc:
                self.model.xfc_weight.grad += torch.addmm(self.model.dummy,out.t(),embed_out,beta=0,alpha=1).to(torch.float32)
            else:
                torch.addmm(self.model.xfc_weight.grad,out.t(),embed_out,beta=1,alpha=1,out=self.model.xfc_weight.grad)
        
        return self.loss

# changed to handle hybrid data-model parallel architecture
class FullPredictor():
    def __init__(self, K=5):
        self.K = K

    def __call__(self, loss_model, model: GenericModel, dataloader: DataLoader):
        datalen = len(dataloader.dataset)
        if model.rank == 0:
          data = np.zeros((datalen, self.K))
          inds = np.zeros((datalen, self.K)).astype(np.int32)
          indptr = np.arange(0, datalen*self.K+1, self.K)
        ctr = 0; numy = model.numy
        model.eval()

        xfcz = model.xfc_batch_size
        if model.world_size > 1:
          self.top_data_gather_list = [torch.empty(xfcz, self.K, dtype = torch.float16 if model.fp16xfc else torch.float32, device=model._target_device)
                     for _ in range(model.world_size)]
          self.top_inds_gather_list = [torch.empty(xfcz, self.K, dtype=torch.long, device=model._target_device)
                     for _ in range(model.world_size)]

        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataloader, desc="Evaluating",disable=(model.rank != 0))):
                batch_data = model.batch_to_device(batch_data, model._target_device)

                bsz = batch_data['batch_size']
                with torch.cuda.amp.autocast(enabled=model.fp16encoder):
                  embed_out = model.embed(batch_data['xfts']) 

                embed_out = loss_model.gather_embed(embed_out, batch_data)

                start_bs = 0
                end_bs = xfcz
                if end_bs > bsz: # partial batch
                  end_bs = bsz
                  xfcz = end_bs - start_bs
                while (end_bs <= bsz):
                    # Do forward pass through XFC layer
                    if model.default_impl:
                      out = loss_model.xfc_forward(embed_out[start_bs:end_bs,:])
                    else:
                      out = model.outsoft[0:end_bs-start_bs,:]
                      loss_model.xfc_forward(embed_out[start_bs:end_bs,:], out)

                    # Compute top predictions
                    # torch.topk is very slow on large tensors, 10+x slowers than matmul! 
                    if self.K <= 5: # torch.max on large tensor is 20x faster than torch.topk, looping 5x is still 4x faster!
                        top_data,top_inds = torch.max(out,1)
                        batch_index = torch.arange(xfcz,device=model._target_device)
                        min_value = torch.tensor(torch.finfo(out.dtype).min,device=model._target_device,dtype=out.dtype)
                        top1_inds = top_inds
                        for i in range(self.K-1):
                          out.index_put_(indices=(batch_index,top1_inds),values=min_value) # replace max values
                          top1_data,top1_inds = torch.max(out,1)  # compute next max
                          top_data = torch.dstack((top_data,top1_data))
                          top_inds = torch.dstack((top_inds,top1_inds))
                    else:
                        top_data, top_inds = torch.topk(out, self.K)

                    # Gather top predictions from all nodes and compute top again
                    if model.world_size > 1:
                        if xfcz != model.xfc_batch_size: # partial batch
                            self.top_data_gather_list = [torch.empty(xfcz, self.K, dtype=top_data.dtype, device=model._target_device)
                              for _ in range(model.world_size)]
                            self.top_inds_gather_list = [torch.empty(xfcz, self.K, dtype=top_inds.dtype, device=model._target_device)
                              for _ in range(model.world_size)]

                        #adjust for model-parallel labels
                        top_inds += model.rank*model.numy_per_gpu  # includes padded labels
                        if model.rank == model.world_size - 1:
                            top_inds[top_inds>=model.numy] = model.numy-1 # remove padded labels
 
                        #all_gather topk from all nodes (gather to one node is sufficient but nccl doesn't support it yet)
                        #print(top_data[0],top_inds[0], top_data.size(),top_inds.size())
                        # TODO: optimize to one all_gather
                        dist.all_gather(self.top_data_gather_list, top_data) 
                        dist.all_gather(self.top_inds_gather_list, top_inds)    
                        top_data_aggr = torch.hstack([self.top_data_gather_list[i] for i in range(model.world_size)])
                        top_data, top_inds_temp = torch.topk(top_data_aggr, self.K)
                        top_inds_aggr = torch.hstack([self.top_inds_gather_list[i] for i in range(model.world_size)])    
                        top_inds = torch.gather(top_inds_aggr, 1, top_inds_temp)
                        #print(top_data_final[0],top_inds_final[0])
                    else:
                        top_inds[top_inds>=model.numy] = model.numy-1 # remove padded label


                    if model.rank == 0:
                        data[ctr:ctr+xfcz] = top_data.float().detach().cpu().numpy()
                        inds[ctr:ctr+xfcz] = top_inds.detach().cpu().numpy()
                        ctr += xfcz

                    start_bs = end_bs
                    end_bs += xfcz
                    if start_bs < bsz and end_bs > bsz:
                        end_bs = bsz
                        xfcz = end_bs - start_bs

        if model.rank == 0:
            return csr_matrix((data.ravel(), inds.ravel(), indptr), (datalen, numy))
        else:
            return None

class PrecEvaluator():
    def __init__(self, model: GenericModel, dataloader, predictor, filter_mat = None, K=5, metric='P', inv_prop=-1):
        self.K = K
        self.metric = metric
        self.dataloader = dataloader
        self.predictor = predictor
        self.filter_mat = filter_mat
        self.inv_prop = inv_prop
        self.model = model
        self.best_score = -9999999

    def __call__(self, loss_model, epoch: int = -1, loss: float = -1.0, out_dir: str = None, name: str = ''):
        if self.model.rank == 0:
          print(self.dataloader.dataset.labels.shape,flush=True)
          print(f'Evaluating {name} {["after epoch %d: "%epoch, ": "][name == ""]}', flush=True)
        #self.predictor.K = max(self.predictor.K, 100)
        if (epoch < 0) or (epoch == self.model.epochs - 1):
          self.predictor.K = 100 #100
        else:  # save time on large datasets
          self.predictor.K = 5
        score_mat = self.predictor(loss_model, self.model, self.dataloader)

        if self.model.rank == 0:
          print(score_mat.shape,flush=True)
          if out_dir is not None:
            score_out_file = f'{out_dir}/{[name+"_", ""][name == ""]}score_mat.npz'
            save_npz(score_out_file, score_mat)
          print('Calculating accuracy in rank 0...',flush=True)
          if self.filter_mat is not None:
            _filter(score_mat, self.filter_mat, copy=False)
          res = printacc(score_mat, X_Y=self.dataloader.dataset.labels, K=max(5, self.K), disp=False, inv_prop_=self.inv_prop) 
       
        if self.model.rank == 0 and out_dir is not None:
            out_file = f'{out_dir}/{[name+"_", ""][name == ""]}evaluation.tsv'
            print(f'dumping evaluation in {out_file}')
            if not os.path.exists(out_file):
                print('\t'.join(['epoch', 'time', 'loss', *[f'{metric}@1' for metric in res.index], *[f'{metric}@3' for metric in res.index], *[f'{metric}@{self.K}' for metric in res.index]]), file=open(out_file, 'w'))
            with open(out_file, 'a') as f:
                print('\t'.join([str(epoch), str(datetime.datetime.now()), str("%.4E"%loss), *["%.2f"%val for val in res['1'].values], *["%.2f"%val for val in res['3'].values], *["%.2f"%val for val in res[str(self.K)].values]]), file=f)
        if self.model.rank == 0:
          score = res[str(self.K)][self.metric]
          score_tensor = torch.tensor(score, dtype=torch.float32, device=self.model._target_device)
        else:
          score = 0.0
          score_tensor = torch.tensor(score, dtype=torch.float32, device=self.model._target_device)

        if self.model.world_size > 1:
          dist.broadcast(score_tensor,src=0, async_op = False)
        score = score_tensor.cpu().item()
 
        if score > self.best_score:
            if self.model.rank == 0:
              print(f'Rank {self.model.rank}: found best model with score : {"%.4f"%score}', flush=True)
              print('\t'.join(['epoch', 'time', 'loss', *[f'{metric}@1' for metric in res.index], *[f'{metric}@3' for metric in res.index], *[f'{metric}@{self.K}' for metric in res.index]]))
              print('\t'.join([str(epoch), str(datetime.datetime.now()), str("%.4E"%loss), *["%.2f"%val for val in res['1'].values], *["%.2f"%val for val in res['3'].values], *["%.2f"%val for val in res[str(self.K)].values]]), flush=True)

            self.best_score = score
            if out_dir is not None:
                #print(f'saving best model in {out_dir}')
                self.model.save(out_dir)
        return score


class PreTokBertDataset(torch.utils.data.Dataset):
    def __init__(self, tokenization_folder, X_Y, num_points, max_len, doc_type='trn', iter_mode='pointwise'):
        self.num_points = num_points
        self.max_len = max_len
        self.iter_mode = iter_mode
        self.labels = X_Y
        self.start =  True
        self.tokenization_folder = tokenization_folder
        self.doc_type = doc_type
        self.num_points = num_points
        self.max_len = max_len 
        #self.Y_ii = np.memmap(f"{tokenization_folder}/lbl_input_ids.dat", 
        #                      mode='r', shape=(X_Y.shape[1], max_len), dtype=np.int64)
        #self.Y_am = np.memmap(f"{tokenization_folder}/lbl_attention_mask.dat", 
        #                      mode='r', shape=(X_Y.shape[1], max_len), dtype=np.int64)
            
    def __getitem__(self, index):
        return index 
    
    def get_fts(self, indices, source='point'):
        if self.start: # doing the memmap in init duplicates it across processes when num_workers > 0; OOM for large datasets; somehow doing it here avoids this issue!
           self.start = False
           self.X_ii = np.memmap(f"{self.tokenization_folder}/{self.doc_type}_doc_input_ids.dat",
                              mode='r', shape=(self.num_points, self.max_len), dtype=np.int64)
           self.X_am = np.memmap(f"{self.tokenization_folder}/{self.doc_type}_doc_attention_mask.dat",
                               mode='r', shape=(self.num_points, self.max_len), dtype=np.int64)
        if source == 'point':
            return bert_fts_batch_to_tensor(self.X_ii[indices], self.X_am[indices])
        if source == 'label':
            return bert_fts_batch_to_tensor(self.Y_ii[indices], self.Y_am[indices])
   
    def __len__(self):
        return self.num_points


# changed to handle hybrid data-model parallel architecture
class XCCollator():
    def __init__(self, padded_numy, dataset, my_rank, world_size, accum, xfcz, train, yfull):
        self.numy = padded_numy
        self.dataset = dataset
        self.rank = my_rank
        self.world_size = world_size
        self.startlabel = self.rank*self.numy//world_size
        self.endlabel = (self.rank+1)*self.numy//world_size
        self.test = not train
        self.accum = accum
        self.xfcz = xfcz # xfc batch size with accum
        self.yfull = yfull
    
    def __call__(self, batch):
        bsz = len(batch)
        per_gpu_batch_size = bsz//self.world_size  
        startbs = self.rank*per_gpu_batch_size
        endbs = startbs + per_gpu_batch_size
        full_ids = np.array(batch)
        # input has batch_size/world_size of total input (final partial batches handled after embedding is computed)
        remainder = bsz % self.world_size
        if remainder > 0:
          if (self.rank < remainder): # spread remainder to first n ranks
            startbs += self.rank
            endbs += self.rank + 1
          else:
            startbs += remainder
            endbs += remainder
        ids = full_ids[startbs:endbs]

        if self.test:
           return {'batch_size': bsz, 'numy': self.numy,  'xfts': self.dataset.get_fts(ids, 'point') } 

        # labels has full batch size but only the partial set of labels that each node is responsible for
        csr_coo  =  self.dataset.labels[full_ids].tocoo()
        #print(self.rank,ids[0],self.dataset.labels[full_ids],csr_coo.col)
        pos_tensor = torch.LongTensor(np.stack((csr_coo.row, csr_coo.col)))
        index_tensor = None

        if self.accum > 1:
            # Compute index tensor to index into 'z' for sliced computation of xfc 
            # TODO: find a cleaner way to extract lengths of various partial batches
            num_xfc_batches = (bsz + self.xfcz - 1)//self.xfcz
            index_tensor = torch.LongTensor(num_xfc_batches+1)
            index_tensor[0] = 0
            start_bs = 0
            end_bs = self.xfcz
            if end_bs > bsz: # partial batch
                end_bs = bsz
            i = 0
            while (end_bs <= bsz):
                temp_ids = full_ids[start_bs:end_bs]
                index_tensor[i+1] = index_tensor[i] + self.dataset.labels[temp_ids].nnz
                pos_tensor[0,index_tensor[i]:index_tensor[i+1]] -= start_bs
                #print(full_ids[0],i,index_tensor[i],len(csr_coo.row),flush=True)
                i += 1
                start_bs = end_bs
                end_bs += self.xfcz
                if start_bs < bsz and end_bs > bsz:
                    end_bs = bsz


        batch_data = {'batch_size': bsz,
                      'numy': self.numy,
                      'z': pos_tensor, #using sparse values instead of 'y'
                      'i': index_tensor,
                      'y': None,
                      #'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy), 
                      'yfull': None,
                      #'ids': torch.LongTensor(ids),
                      'xfts': self.dataset.get_fts(ids, 'point')
                     }
            
            
        if self.yfull:
            batch_data['y'] = csr_to_pad_tensor(self.dataset.labels[full_ids], self.numy//self.world_size)
            #batch_data['yfull'] = torch.zeros(bsz, self.numy+1).scatter_(1, batch_data['y']['inds'], batch_data['y']['vals'])[:, :-1]
                
        return batch_data
