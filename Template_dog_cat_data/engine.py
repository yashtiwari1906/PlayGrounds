import trainer
import common
from datetime import datetime as dt 
import argparse

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to configuration file")
    ap.add_argument("-p", "--path", required=True, help="Path to root directory")
    ap.add_argument("-b", "--batch_size", default=100, type = int,  help="batch_size")
    ap.add_argument("-d", "--device", default="cpu", type=str, help="Whether to perform task on CPU ('cpu') or GPU ('cuda')")
    ap.add_argument("-t", "--task", default="train", type=str, help="Task to perform. Choose between ['train', 'test']")
    ap.add_argument("-o", "--output", default=dt.now().strftime("%d-%m-%Y-%H-%M"), type=str, help="Output directory path")
    ap.add_argument("-l", "--load", default=None, type=str, help="Path to directory containing checkpoint as best_model.pt")
    args = vars(ap.parse_args())
    path = r"Y:\\pytorch\\natural_images"
    #args = {"config": r"C:\Users\yasht\Desktop\template2\configs\main.yaml", "task": "train", "output": r"C:\Users\yasht\Desktop\template2", "load":None}
    trainer = trainer.Trainer(args)
    if args["task"] == "train":
        trainer.train()
    
    elif args["task"] == "test":
        assert args["load"] is not None, "Please provide a checkpoint to load using --load to check test performance"
        trainer.get_test_performance()

    elif args["task"] == "single_test":
        assert os.path.exists(args["file"]), "No wav file found at path provided"
        assert args["load"] is not None, "Please provide a checkpoint to load using --load to check test performance"
        trainer.predict_for_file(args["file"])

    else:
        raise ValueError(f"Unrecognized argument passed to --task: {args['task']}")
    

 

    print("=====================================================================================================")
