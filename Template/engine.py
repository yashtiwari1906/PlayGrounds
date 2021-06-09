import model
from dataset import PrepareDataset
import common

if __name__ == "__main__":
    path = r"Y:\pytorch\data\train.csv"
    batch_size = 100
    args = {"config": r"C:\Users\yasht\Desktop\template_try_deep_learning\configs\main.yaml", "task": "train", "output": r"C:\Users\yasht\Desktop\template_try_deep_learning", "load":None}
    trainer = model.Trainer(args, path, batch_size)
    trainer.train()

    trainer.get_test_performance()

    print("=====================================================================================================")
