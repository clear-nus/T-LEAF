import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", default=False, help="don't use cuda")
    parser.add_argument("--device_id", default=0, type=int, help="gpu id")

    parser.add_argument("--video_dataset_info", type=str, default="./datasets/Action_Recognition/videos/info/")
    parser.add_argument("--video_dataset_path", type=str, default="/home/user/Documents/zf/LinearTemporalLogicEmbedding/dataset/Label_Dataset/")

    parser.add_argument("--logic_dataset_path", type=str, default="./datasets/Action_Recognition/")
    parser.add_argument("--logic_dataset_name", type=str, default="/meta_embedder_dataset/")

    parser.add_argument("--edge_embedder_path", type=str, default="./saved_models/Action_Recognition/edge_embedder/edge_embedder_latest.pt")
    parser.add_argument("--meta_embedder_path", type=str, default="./saved_models/Action_Recognition/meta_embedder/meta_embedder_dataset/meta_embedder_latest")

    parser.add_argument("--model_save_path", type=str, default="./saved_models/Action_Recognition/videos/TCN/")



    parser.add_argument("--ratio",type=float ,default=0.8)
    parser.add_argument("--lr",type=float ,default=0.0001,help="learning rate")
    parser.add_argument("--weight_decay",type=float ,default=0.001,help="")
    parser.add_argument("--batch_size", type=int,default=16)
    parser.add_argument("--input_dim", type=int,default=2048)
    parser.add_argument("--output_dim", type=int,default=512)
    parser.add_argument("--hidden_dim", type=int,default=800)
    parser.add_argument("--num_action", type=int,default=64)
    parser.add_argument("--num_ingredient", type=int,default=155)
    parser.add_argument("--interval", type=int,default=5)
    parser.add_argument("--num_data", type=int,default=500)
    parser.add_argument("--step_avg", default=False, action="store_true", help="use lstm or avg to solve the step frame")
    parser.add_argument("--threshold", type=float, default=0.5, help= " the threshold to judge the output is 1 or 0")
    parser.add_argument("--clip", type=float, default=3.0, help= " the threshold to judge the output is 1 or 0")
    parser.add_argument("--multi_label",default=False, action="store_true", help="use Multilabel or others")
    parser.add_argument("--embedder_loss",default=False, action="store_true", help="embedder loss")


    parser.add_argument("--top_k_acc", type=int, default=5, help="the top k accuracy")
    parser.add_argument("--seed", type=int, default=27, help="random seed for reprobudicibility")
    parser.add_argument("--eval", default=False, action="store_true", help="eval precision ")
    parser.add_argument("--macro", default=False, action="store_true", help="precision parameter ")


    parser.add_argument("--checker_loss",default=False, action="store_true", help="checker loss")

    parser.add_argument("--meta_train",default=False, action="store_true", help="train meta_embedder")
    parser.add_argument("--memory_predictor",default=False, action="store_true", help="using lstm to generate the output")

    parser.add_argument("--lamda", default=3, type=int, help="tradeoff factor")

    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args