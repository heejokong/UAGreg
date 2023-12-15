import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='[PyTorch] OPEN-SET SSL')
    ## CONFIGURATIONS
    parser.add_argument('--trainer', type=str, default='sup', help='mode to train')
    parser.add_argument('--eval_only', type=int, default=0, help='1 if evaluation mode ')
    # COMPUTATIONAL CONFIG
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--seed', default=612, type=int, help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    #
    parser.add_argument("--use_amp", action="store_true", default=False, help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O2",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    # SAVE CONFIG
    parser.add_argument('--root', default='../data', type=str, help='path to data directory')
    parser.add_argument('--out', default='result', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

    ## HYPER-PARAMETERS FOR TRAINING
    # MODEL CONFIG
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'wideresnet'])
    # DATASET CONFIG
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='dataset name')
    parser.add_argument('--num_classes', type=int, default=6, help='total number of classes for training')
    parser.add_argument('--ratio', type=float, default=0.5, help='proportion of class distribution mismatch')
    parser.add_argument('--num_labeled', type=int, default=400, help='number of labeled data per each class')
    parser.add_argument('--num_unlabeled', type=int, default=20000, help='total number of labeled data')
    parser.add_argument('--num_val', type=int, default=50, help='number of validation data per each class')
    parser.add_argument("--expand_labels", action="store_true", default=True, help="expand labels to fit eval steps")
    # 
    parser.add_argument('--label_biased', default=False, action='store_true')
    parser.add_argument('--label_modify', default=False, action='store_true')
    # OPTIMIZER CONFIG
    parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam', 'lars'], help='optimizer name')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['cosine', 'step_decay'], help='scheduler name')
    parser.add_argument('--warm_up', default=0, type=float, help='warm_up epochs (unlabeled data based)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    # OTHERS
    parser.add_argument('--total_step', default=2**19, type=int, help='number of total steps to run')
    parser.add_argument('--eval_step', default=1024, type=int, help='number of eval steps to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
    parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
    # EMA CONFIG
    parser.add_argument('--use_ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema_decay', default=0.999, type=float, help='EMA decay rate')
    # FIXMATCH CONFIG
    parser.add_argument('--start_u', default=0, type=float, help='start epoch for fixmatch loss')
    parser.add_argument('--lambda_u', default=1., type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--conf_th', default=0.95, type=float, help='')
    # COMATCH CONFIG
    parser.add_argument('--graph_th', default=0.8, type=float, help='')
    parser.add_argument('--lambda_g', default=1., type=float, help='coefficient of graph regularization loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    # PROPOSED CONFIG
    parser.add_argument('--temp_o', default=1.0, type=float, help='')
    parser.add_argument('--momentum', default=0.9, type=float, help='')
    # OTHER CONFIG
    # parser.add_argument('--', default=, type=, help='')

    args = parser.parse_args()
    return args
