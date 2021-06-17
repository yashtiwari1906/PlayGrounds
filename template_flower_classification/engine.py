import trainer
import common
from datetime import datetime as dt 
import argparse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required = True, help = "path to configuration file")
    ap.add_argument("-p", "--path", required=True, help = "path to data")
    ap.add_argument("-b", "--batch_size", default = 100, type=int, help = "batch_size")
    ap.add_argument("-d", "--device", default = "cuda", type=str, help = "cpu or gpu")
    ap.add_argument("-t", "--task", default = "TRAIN", type=str, help = "perfor on train ('TRAIN') or test('TEST')")
    ap.add_argument("-o", "--output", default = dt.now().strftime("%d-%m-%Y-%H-%M"), type=str, help = "path to output directory")
    ap.add_argument("-l", "--load", default = None, type=str, help = "ath to directory containing checkpoint as best_model.pt")
    args = vars(ap.parse_args())

    trainer = trainer.Trainer(args)

    if args["task"] == "TRAIN":
        trainer.train()

    elif args["task"] == "TEST":
        assert args["load"] is not None, "Please provide a checkpoint to load using --load to check test performance"
        trainer.get_test_performance()
    else:
        raise ValueError(f"Unrecognized argument passed to --task: {args['task']}")

    #python engine.py -c C:\Users\yasht\Desktop\flower_classification\configs\main.yaml -p Y:\pytorch\flower_classification\flower_data\flower_data\train -o C:\Users\yasht\Desktop\flower_classification

