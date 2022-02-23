import argparse

parser = argparse.ArgumentParser(description="Training and evaluation of the model")

parser.add_argument(
    "-m",
    "--base_model",
    default="facebook/wav2vec2-xls-r-300m",
    required=False,
    type=str,
    help="base model from HuggingFace",
)

parser.add_argument(
    "-c",
    "--collection",
    type=str,
    choices=["IEMOCAP", "IEMOCAP_Norm_MinMax", "IEMOCAP_Norm_Std"],
    required=False,
    default="IEMOCAP",
    help="MongoDB Collection for the Database. It can be 'IEMOCAP', 'IEMOCAP_Norm_MinMax' or 'IEMOCAP_Norm_Std'",
)
parser.add_argument(
    "-s",
    "--shutdown",
    type=bool,
    choices=[False, True],
    default=False,
    required=False,
    help="Shut down the computer after the evaluation has finished",
)

parser.add_argument(
    "-d", "--dataset_size", type=int, required=False, help="size of the dataset"
)

parser.add_argument(
    "-b", "--batch_size", type=int, required=False, default=2, help="size of the batch for training"
)

parser.add_argument(
"--save_step", type=int, required=False, default=20, help="steps for saving a checkpoint while training"
)
parser.add_argument(
"--eval_step", type=int, required=False, default=20, help="steps for evaluation while training"
)
parser.add_argument(
    "-e",
    "--no_evaluate",
    action= 'store_false',
    dest='TEST',
    help="Does not evaluate the model",
)

parser.add_argument(
    "-t",
    "--no_train",
    action= 'store_false',\
    dest='TRAIN',
    help="Does not train the model",
)


args = parser.parse_args()
