import torch
from res import * 
from dl_base import *
import numpy as np
import scipy.sparse as sp
import os
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import RobertaModel, AutoModel, AutoConfig
from collections import OrderedDict
#from torch.profiler import profile, record_function, ProfilerActivity

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    #print(torch.utils.data.get_worker_info())

# Chunk the data into slices and fix randomness within a slice
class MySubsetRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, total, num_slices, slice_size, generator=None):
        self.total = total
        self.part = slice_size
        self.num_slices = num_slices
        self.generator = generator
        self.start = 0
        self.rem = total - slice_size*num_slices
    def __iter__(self):
        for i in range(self.num_slices-1):
            yield from (self.start + torch.randperm(self.part, generator=self.generator)).tolist()
            self.start += self.part
        yield from (self.start + torch.randperm(self.part+self.rem, generator=self.generator)).tolist()
        self.start = 0
    def __len__(self):
        return self.total   

def start(device, ngpus_per_node, args):
  # set seed
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)

  torch.cuda.set_device(device)
  nb_id = device
  # initialize the process group
  if args.world_size > 1: 
    if args.world_size > ngpus_per_node:  # multi node, multi gpu training
        args.rank = args.rank * ngpus_per_node + device
    else:                                 # single node, multi gpu training
        args.rank = device
    my_rank = args.rank
    dist.init_process_group("nccl", init_method=args.dist_url, rank=my_rank, world_size=args.world_size)
  else:                                  # single node, single gpu training
    args.rank = my_rank = 0

#   root_dir = '/home/jainvidit/xc'
  root_dir = '/home/t-japrakash/xc'
  if not os.path.exists(root_dir):
      root_dir = '/home/jainvidit/xc'
    
    
  results_dir = f'./Results/Bert-XC/'
  os.makedirs(results_dir, exist_ok=True)
  dataset=args.dataset

  device = f'cuda:{nb_id}'
#   if args.world_size > 1:
#     expname = f'BertDXML2-distributed{args.world_size}'
#   else:
#     expname = f'BertDXML2-{nb_id}'
  expname = args.expname

  torch.cuda.set_device(device)


  # Load Data
  DATA_DIR = f'{root_dir}/Datasets/{dataset}'
  if os.path.isfile(f'{DATA_DIR}/trn_X_Y_sparse_mat.npz'):
    trn_X_Y = sp.load_npz(f'{DATA_DIR}/trn_X_Y_sparse_mat.npz')
  else:
    trn_X_Y = read_sparse_mat(f'{DATA_DIR}/trn_X_Y.txt')
  if args.pre_tok:
    if my_rank == 0: # only rank 0 does eval 
      if os.path.isfile(f'{DATA_DIR}/tst_X_Y_sparse_mat.npz'):
        tst_X_Y = sp.load_npz(f'{DATA_DIR}/tst_X_Y_sparse_mat.npz')
      else:
        tst_X_Y = read_sparse_mat(f'{DATA_DIR}/tst_X_Y.txt')
      tst_shape_0, tst_shape_1 = tst_X_Y.shape[0], tst_X_Y.shape[1]
    else: # just get shape
      if os.path.isfile(f'{DATA_DIR}/tst_X_Y_sparse_mat.npz'):
        tst_X_Y = sp.load_npz(f'{DATA_DIR}/tst_X_Y_sparse_mat.npz')
        tst_shape_0, tst_shape_1 = tst_X_Y.shape[0], tst_X_Y.shape[1]
      else:
        tst_X_Y = read_sparse_mat(f'{DATA_DIR}/tst_X_Y.txt')
        fo = open(f'{DATA_DIR}/tst_X_Y.txt', "r")
        line = fo.readline()
        tst_shape_0, tst_shape_1 = map(int,line.split())
        fo.close()
      tst_X_Y = None
  else:
      tst_X_Y = read_sparse_mat(f'{DATA_DIR}/tst_X_Y.txt')
      tst_shape_0, tst_shape_1 = tst_X_Y.shape[0], tst_X_Y.shape[1]
  #print(tst_shape_0,tst_shape_1,flush=True)

  if "Amazon" in dataset: A = 0.6; B = 2.6
  elif "Wiki" in dataset: A = 0.5; B = 0.4
  else : A = 0.55; B = 1.5
  inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, A, B)

  temp = np.fromfile('%s/trn_filter_labels.txt'%(DATA_DIR), sep=' ').astype(int)
  temp = temp.reshape(-1, 2).T
  trn_filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), trn_X_Y.shape).tocsr()

  temp = np.fromfile('%s/tst_filter_labels.txt'%(DATA_DIR), sep=' ').astype(int)
  temp = temp.reshape(-1, 2).T
  tst_filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), (tst_shape_0,tst_shape_1)).tocsr()


  if 'roberta' in args.tf: tokenizer_type = 'roberta-base'
  elif 'bert' in args.tf: tokenizer_type = 'bert-base-uncased'
  elif 'custom' in args.tf: tokenizer_type = 'custom'
  elif 'MiniLM' in args.tf: tokenizer_type = 'miniLM_L6'
  else:
      tokenizer_type = args.tf
      print("Potentially unsupported tokenizer_type ",args.tf)
      if 'MiniLM' in args.tf:
        tokenizer_type = 'roberta-base'

  padded_numy = trn_X_Y.shape[1]
  if (padded_numy % args.world_size != 0):
        padded_numy = args.world_size*((trn_X_Y.shape[1] + args.world_size -1)//args.world_size)
        if my_rank == 0:
          print("Rounding numy to",padded_numy)

  numy_per_gpu = padded_numy // args.world_size
  if (numy_per_gpu % 16 != 0):
        padded_numy = args.world_size*16*((numy_per_gpu + 15)//16)
        numy_per_gpu = padded_numy // args.world_size
        if my_rank == 0:
          print("Rounding numy further to",padded_numy)
  if my_rank == 0:
    print("Final numy_per_gpu: ",numy_per_gpu)

  #Dataloaders
  num_points = trn_X_Y.shape[0]
  if args.slice > 1:
    # all elements of a batch should fit in a slice
    slice_size = (num_points//(args.batch_size*args.world_size*args.slice))*args.batch_size*args.world_size
  else:
    slice_size = num_points 
  start_label = my_rank*numy_per_gpu
  end_label = (my_rank+1)*numy_per_gpu
  if args.pre_tok:
      # restrict the train dataset to only the labels that this rank is responsible for
      start_row = 0
      end_row = slice_size
      trn_X_Y_list = []
      for i in range(args.slice-1):
        trn_X_Y_list.append(trn_X_Y[start_row:end_row].tocsc()[:,start_label:end_label].tocsr()) 
        start_row = end_row
        end_row += slice_size
      end_row = num_points
      trn_X_Y_list.append(trn_X_Y[start_row:end_row].tocsc()[:,start_label:end_label].tocsr()) 
      trn_dataset = PreTokBertDataset(f'{DATA_DIR}/{tokenizer_type}-{args.maxlen}', trn_X_Y_list, num_points, args.maxlen, doc_type='trn')
      tst_dataset = PreTokBertDataset(f'{DATA_DIR}/{tokenizer_type}-{args.maxlen}', [tst_X_Y], tst_shape_0, args.maxlen, doc_type='tst')
      #tst_dataset = PreTokBertDataset(f'{DATA_DIR}/{tokenizer_type}-{args.maxlen}', [tst_X_Y], tst_shape_0, args.maxlen, doc_type='trn')
  else:
      OUT_DIR = f'{results_dir}/{dataset}'
      Y = [x.strip() for x in open(f'{DATA_DIR}/raw/Y.txt').readlines()]
      trnX = [x.strip() for x in open(f'{DATA_DIR}/raw/trn_X.txt').readlines()]
      tstX = [x.strip() for x in open(f'{DATA_DIR}/raw/tst_X.txt').readlines()]
      #trn_X_Y = trn_X_Y.tocsc()[:,start_label:end_label].tocsr()
      trn_dataset = BertDataset(trnX, Y, [trn_X_Y], maxsize=args.maxlen, tokenizer_type=args.tf,data_root_dir=OUT_DIR)
      tst_dataset = BertDataset(tstX, Y, [tst_X_Y], maxsize=args.maxlen, tokenizer_type=args.tf,data_root_dir=OUT_DIR)

  gbsz = args.batch_size*args.world_size # everyone gets all labels
  num_workers = 4
  trn_loader = torch.utils.data.DataLoader( 
    trn_dataset,
    sampler=MySubsetRandomSampler(num_points, args.slice, slice_size) if args.slice > 1 else None,
    batch_size=gbsz,
    num_workers=num_workers,
    collate_fn=XCCollator(padded_numy, trn_dataset, my_rank, args.world_size, args.slice, slice_size, args.accum, gbsz//args.accum, True, args.default_impl),
    worker_init_fn=seed_worker, # need workers to use same seed so that batches are coordinated
    persistent_workers = False if ((args.slice > 1)  or (num_workers < 1)) else True,
    shuffle=False if args.slice > 1 else True,
    pin_memory=False)

  tst_loader = torch.utils.data.DataLoader( 
    tst_dataset, 
    batch_size=gbsz,
    num_workers=num_workers,
    collate_fn=XCCollator(padded_numy, tst_dataset, my_rank, args.world_size, args.slice, slice_size, args.accum, gbsz//args.accum, False, args.default_impl),
    worker_init_fn=seed_worker, # need workers to use same seed so that batches are coordinated
    persistent_workers = False if ((args.slice > 1)  or (num_workers < 1)) else True,
    shuffle=False,
    pin_memory=False)

  # Pre-trained custom Transformer Input layer
  
  try:
  #encoder = TransformerInputLayer(AutoModel.from_config(AutoConfig.from_pretrained(args.tf),add_pooling_layer=False), 'mean') # randomly initialized model
      encoder_ = AutoModel.from_pretrained(args.tf,add_pooling_layer=False)
  except:
      encoder_ = AutoModel.from_pretrained(args.tf)

    
  # whether or not to use ngame pretrained encoder (which is M1 in the ngame paper) as initialization.
  if args.use_ngame_encoder: 
    path_to_ngame_model = f"/blob/ngame_pretrained_models/{args.dataset}/state_dict.pt"
    print("Using NGAME pretrained encoder. Loading from {}".format(path_to_ngame_model))

    new_state_dict = OrderedDict()
    old_state_dict = torch.load(path_to_ngame_model, map_location="cpu")

    for k, v in old_state_dict.items():
        name = k.replace("embedding_labels.encoder.transformer.0.auto_model.", "") # for Academic datasets
        # name = k.replace("encoder.encoder.transformer.0.auto_model.", "") # for EPM-20M
        new_state_dict[name] = v

    print(encoder_.load_state_dict(new_state_dict, strict=True))  
  encoder = TransformerInputLayer(encoder_, 'mean')
            
  if args.bottleneck_dims > 0:
    # modules = [encoder, nn.Dropout(args.dropout), FP32Linear(encoder.transformer.config.hidden_size, args.bottleneck_dims, bias=False)] 
    # below version works better than above
    modules = [encoder, FP32Linear(encoder.transformer.config.hidden_size, args.bottleneck_dims, bias=False), nn.Dropout(args.dropout) ] 
    # multiple layers doesn't help 
    # modules = [encoder, FP32Linear(encoder.transformer.config.hidden_size, 4*args.bottleneck_dims, bias=False), FP32Linear(4*args.bottleneck_dims, args.bottleneck_dims, bias=False), nn.Dropout(args.dropout) ] 
  else:
    modules = [encoder, nn.Dropout(args.dropout)] 
    args.bottleneck_dims = encoder.transformer.config.hidden_size

  model = GenericModel(my_rank, args, trn_X_Y.shape[1], numy_per_gpu, args.batch_size, modules, device=device, name=expname, out_dir=f'{results_dir}/{dataset}/{expname}')
  model = model.cuda()
  #model.load()

  if args.world_size > 1:
    if not args.nosync:
      model.embed = DDP(model.embed, device_ids=[my_rank%ngpus_per_node])

  if args.default_impl:
      trn_loss = BCELossMultiNodeDefault(model)
  else:
      trn_loss = BCELossMultiNode(model)
  '''
  if args.world_size == 1:
    if args.default_impl:
      trn_loss = BCELossMultiNodeDefault(model)
    else:
      trn_loss = BCELossMultiNode(model)
      #trn_loss = BCELoss(model)
  else:
    if args.default_impl:
      trn_loss = BCELossMultiNodeDefault(model)
    else:
      trn_loss = BCELossMultiNode(model)
  '''

  if args.init_weights:
      totlen = trn_X_Y.shape[1] #numy
      Y_ii = np.memmap(f"{DATA_DIR}/{tokenizer_type}-{args.maxlen}/lbl_input_ids.dat",
                              mode='r', shape=(totlen, args.maxlen), dtype=np.int64)
      Y_am = np.memmap(f"{DATA_DIR}/{tokenizer_type}-{args.maxlen}/lbl_attention_mask.dat",
                              mode='r', shape=(totlen, args.maxlen), dtype=np.int64)
      if my_rank == args.world_size - 1:
          Y_len = totlen - numy_per_gpu*my_rank
      else:
          Y_len = numy_per_gpu
      start = numy_per_gpu*my_rank
      model.eval()
      with torch.no_grad():
         for i in range(start, start+Y_len, args.batch_size):
            endlen = i + args.batch_size
            if endlen > start+Y_len:
                endlen=start+Y_len
            maxlen = Y_am[i:endlen].sum(axis=1).max()
            #maxlen = args.maxlen
            batch_input = {'input_ids': torch.from_numpy(Y_ii[i:endlen, :maxlen]),
                       'attention_mask': torch.from_numpy(Y_am[i:endlen, :maxlen])}
            for key in batch_input:
              batch_input[key] = batch_input[key].cuda()
            embed_out = model.embed(batch_input)
            embed_norm = torch.norm(embed_out,dim=1,keepdim=True)
            model.xfc_weight.data[i-start:endlen-start, :] = (embed_out/embed_norm)
            #model.xfc_weight.data[i-start:endlen-start, :] = model.embed(batch_input)
      del batch_input, Y_ii, Y_am

  model.epochs = args.epochs
  predictor = FullPredictor()
  evaluator = PrecEvaluator(model, tst_loader, predictor, filter_mat=tst_filter_mat, inv_prop=inv_prop)
  #evaluator(trn_loss,out_dir=None, name=model.name)
  if args.infer:
      model.load()
      evaluator(trn_loss,out_dir=None, name=model.name)
  else:
      model.fit(trn_loader, trn_loss, 
          xfc_optimizer_class = apex.optimizers.FusedSGD,
          xfc_optimizer_params= {'lr': args.lr1, 'momentum': args.mo, 'weight_decay': args.wd1,  'set_grad_none': True},  
          tf_optimizer_class = apex.optimizers.FusedAdam,
          tf_optimizer_params= {'lr': args.lr2, 'eps': 1e-06, 'set_grad_none': True, 'bias_correction': True, 'weight_decay': args.wd2},
          epochs = args.epochs, warmup_steps = args.warmup, explore_steps = args.explore,
          evaluator=evaluator,
          evaluation_epochs=5)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='BERT XFC')
    parser.add_argument('--dataset', type=str, default='LF-Wikipedia-500K', metavar='N',
                        help='name of the dataset')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='Add unit norm constraint for xfc weights')
    parser.add_argument('--custom-cuda', action='store_true', default=False,
                        help='Use custom_cuda kernels for fp16 training. Please ensure the custom kernels are optimized for the matmul sizes!!!')
    parser.add_argument('--default-impl', action='store_true', default=False,
                        help='Use default implemenation -- to get a sense of how much perf optimization gains over the default approach')
    parser.add_argument('--init-weights', action='store_true', default=False,
                        help='Initialize weights for xfc layer using label embeddings of encoder; needs pre_tok')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input per GPU batch size for training (default: 8)')
    parser.add_argument('--slice', type=int, default=1,
                        help='Slice the dataset for perf (default: 6)')
    parser.add_argument('--freeze-epoch', type=int, default=-1, # preliminary, unoptimized version -- TBD: compute embeddings once, and remove one backward matmul
                        help='Freeze encoder starting from freeze-epoch (default: no-freezing) ')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--expname', type=str, default='debug', metavar='N',
                        help='Name of exp')
    parser.add_argument('--device', type=int, default=None, metavar='N',
                        help='Single device training (default: None)')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training; replace localhost with IP of rank 0 for multinode training')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='Rank of node used in training (default: 0, should range from 0 to world-size -1)')
    parser.add_argument('--world-size', type=int, default=1, metavar='N',
                        help='Number of nodes used in training (default: 1)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='seed (default: 42)')
    parser.add_argument('--bottleneck-dims', type=int, default=0, metavar='N',
                        help='bottleneck before fc layer (default: 0 means no bottleneck)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout prob (default: 0.5)')
    parser.add_argument('--lr1', type=float, default=0.0025, metavar='LR',
                        help='learning rate for xfc layer (default: 0.025)')
    parser.add_argument('--lr2', type=float, default=1e-5, metavar='LR',
                        help='learning rate for encoder (default: 1e-4)')
    parser.add_argument('--mo', type=float, default=0.9, metavar='MO',
                        help='momentum for xfc layer (default: 0.9)')
    parser.add_argument('--wd1', type=float, default=0.01, metavar='WD',
                        help='weight decay for xfc layer (default: 0.01)')
    parser.add_argument('--wd2', type=float, default=0.01, metavar='WD',
                        help='weight decay for encoder (default: 0.01)')
    parser.add_argument('--fp32encoder', action='store_true', default=False,
                        help='enable fp32 for encoder (default fp16)')
    parser.add_argument('--fp16xfc', action='store_true', default=False,
                        help='enable fp16 for xfc layer (default fp32)')
    parser.add_argument('--nosync', action='store_true', default=False,
                        help='Skip allreduce of gradients for encoders, surprising this works fine!')
    parser.add_argument('--noloss', action='store_true', default=False,
                        help='Skip loss computation (only accuracy is valid), saving memory/compute at extreme scale')
    parser.add_argument('--checkpoint-resume', type=str, default='',
                        help='DIR to checkpoint each epoch/resume from most recent checkpoint')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='Perform inference of pre-trained model')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Add bias to classifiers')
    parser.add_argument('--warmup', type=int, default=140000, metavar='N',
                        help='number of steps for warmup (default: 140000)')
    parser.add_argument('--explore', type=int, default=0, metavar='N',
                        help='number of steps for explore (default: 0)')
    parser.add_argument('--accum', type=int, default=1, metavar='N',
                        help='gradient accumulation steps in XFC layer to save memory (default: 1)')
    parser.add_argument('--pre-tok', action='store_true', default=False,
                        help='Use pre tokenized .dat files for dataset')
    parser.add_argument('--maxlen', type=int, default=32, metavar='N',
                        help='max seq length for transformer')
    parser.add_argument('--tf', type=str, default='distilbert-base-uncased', metavar='N',
                        help='encoder transformer type')
    parser.add_argument('--use-ngame-encoder', action='store_true', default=False,
                        help='Use NGAME pretrained encoder as initialization point for trainings')
    args = parser.parse_args()

    if args.bias and (args.custom_cuda or args.default_impl):
        print("XFC bias not supported yet for default_impl or custom_cuda kernels: TBD")
        exit(-1)
    if (not args.pre_tok) and (args.tf == "custom" or args.world_size > 1 or args.slice > 1 or args.init_weights):
        print("args.tf == custom or args.world_size > 1 or args.slice > 1 or args.init_weights requires args.pre_tok for now")
        print("Use python -u CreateTokenizedFiles.py --data-dir Datasets/XXX --max-length YYY--tokenizer-type ZZZ to pretokenize data")
        exit(-1)

    print(args)
 
    if args.device is not None:
        print("Specific device chosen. Starting single device training")
        start(args.device, 1, args)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size *= ngpus_per_node
        if args.batch_size*args.world_size % args.accum != 0:
            print("Code currently expects  args.accum cleanly divides args.batch_size*args.world_size")
            exit(-1)
        print("Starting multi device training on ", args.world_size, " devices")
        mp.spawn(start, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)

if __name__ == '__main__':
     main()
