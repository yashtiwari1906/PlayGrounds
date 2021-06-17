
import os
import common
from model import ResnetModel
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataset import FlowerDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(self, args):
        self.path = args["path"]
        self.batch_size = args["batch_size"]
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)
        self.model = ResnetModel().to(self.device)
        files, labels = self.generate_files(self.path)
        train_files, train_labels, val_files, val_labels = train_test_split(files, labels, test_size = 0.15, random_state = 42)
        train_dataset, val_dataset = FlowerDataset(self.path, train_files, train_labels, task = "TRAIN"), FlowerDataset(self.path, val_files, val_labels, task = "TEST")
        self.train_loader, self.val_loader = DataLoader(train_dataset, batch_size= self.batch_size, shuffle = True), DataLoader(val_dataset, batch_size= self.batch_size, shuffle = True)
        self.optim = torch.optim.SGD(self.model.parameters(), lr = self.config["model"]["optim_lr"], weight_decay = 0.005, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max = self.config["epochs"] - self.config["warmup_epochs"], eta_min = 0.0, last_epoch = -1)
        self.warmup_epochs = self.config.get("warmup_epochs", 0)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["model"]["optim_lr"] - 1e-12) / self.warmup_epochs 
        self.done_epochs = 1
        self.metric_best = float("-inf")
        run = wandb.init(project = "project-flower-classification")
        self.logger.write(f"Wandb run: {run.get_url()}", mode = "info")
        if args["load"] is not None:
            self.load_model(args["load"])

    def generate_files(self, path):
        labels = []
        file_names = []
        list_labels = os.listdir(path)
        print("generating list of files for dataset.....")
        for label in tqdm(list_labels):
            files = [label+"//"+ x for x in os.listdir(path+"//"+label)]
            file_names.extend(files)
            labels.extend([label]*len(files))

        assert len(labels) == len(file_names), "labels and file_names are not equal in length"
        return file_names, labels

    def loss_fn(self, preds, labels):
        error = nn.CrossEntropyLoss()
        return error(preds, labels)

    def compute_accuracy(self, loader):
        correct, total = 0, 0
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.view(self.batch_size, 3, 224, 224))
            images =  images.to(self.device)
            outputs = self.model(images)
            predicted = torch.max(outputs.data, 1)[1].detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            correct += (predicted == labels).sum()
            total+=len(labels)
            common.progress_bar(status = "", progress = (i+1)/len(loader))
        accuracy = 100 * correct / float(total)
        common.progress_bar(status = "[accuracy] {:.4f}".format(accuracy), progress = 1.0)
        return accuracy

    def train_on_batch(self, batch):
        self.model.train()
        images, labels = batch
        images, labels = Variable(images.size(self.batch_size, 3, 224, 224)), Variable(labels)
        images, labels = images.to(self.device), labels.to(self.device)
        output = self.model(images)
        self.optim.zero_grad()
        loss = self.loss_fn(output, labels)
        loss.backward()
        self.optim.step()
        return {"Loss": loss.data}

    def infer_on_batch(self, batch):
        self.model.eval()
        images, labels = batch 
        images, labels = Variable(images.view(self.batch_size, 3, 224, 224)), Variable(labels)
        images, labels = images.to(self.device), labels.to(self.device)
        with torch.no_grad():
            output = self.model(images)
            loss = self.loss_fn(output, labels)
        return {"Loss": loss.data}

    def save_model(self, epoch, metric):
        state = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metric": metric, 
            "epoch": epoch
        }
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_model(self, path):
        if not os.path.exists(os.path.join(self.args["load"], "best_model.pt")):
            raise NotImplementedError(f"Could not find saved model 'best_model.pt' at {self.args['load']}")
        else:
            state = torch.load(os.path.join(self.args["load"], "best_model.pt"), map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.optim.load_state_dict(state["optim"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.done_epochs = state["epoch"]
            self.logger.show(f"Succcessfully loaded model from {path}", mode = "info")

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 + epoch * self.warmup_rate
        else:
            self.scheduler.step()

    def get_test_performance(self):
        test_meter = common.AverageMeter()
        for idx in range(len(self.val_loader)):
            batch = self.val_loader[idx]
            test_metrics = self.infer_on_batch(batch)
            test_meter.add(test_metrics)
            common.progress_bar(status = test_meter.return_msg(), progress = (idx+1)/len(self.val_loader))

        common.progress_bar(status = test_meter.return_msg(), progress = 1.0)
        self.logger.record("Computing Accuracy", mode = "info")
        accuracy = self.compute_accuracy(self.val_loader)
        self.logger.record(test_meter.return_msg() + "[accuracy] {:.4f}".format(accuracy), mode = "test")

    def train(self):
        print()
        for epoch in range(max(self.done_epochs, 1), self.config["epochs"]+1):
            self.logger.record(f"Epoch {epoch}/{self.config['epochs']}", mode = "train")
            train_meter = common.AverageMeter()
            for idx, batch in enumerate(self.train_loader):
                train_metrics = self.train_on_batch(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train Loss": train_metrics["loss"]})
                common.progress_bar(status = train_meter.return_msg(), progress = (idx+1)/len(self.train_loader))
            common.progress_bar(status = train_meter.return_msg(), progress = 1.0)
            self.logger.record(f"Epoch {epoch}/{self.config['epoch']} Computing accuracy", mode = "train")
            accuracy = self.compute_accuracy(self.train_loader)
            wandb.log({"Train Accuracy": accuracy, "Epoch": epoch})
            self.logger.write(train_meter.return_msg() + f" [accuracy] {accuracy}", mode="train")
            self.adjust_learning_rate(epoch)

            if epoch % self.config["eval_every"] == 0:
                self.logger.record(f"Epoch {epoch}/{self.config['epochs']}", mode='val')
                val_meter = common.AverageMeter()
                for idx, batch in enumerate(self.val_loader):
                    val_metrics = self.infer_on_batch(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(status=val_meter.return_msg(), progress=(idx+1)/len(self.val_loader))

                common.progress_bar(status=val_meter.return_msg(), progress=1.0)
                self.logger.record(f"Epoch {epoch}/{self.config['epochs']} Computing accuracy", mode='val')
                val_accuracy = self.compute_accuracy(self.val_loader)
                wandb.log({"Val loss": val_meter.return_metrics()["Loss"], "Val accuracy": val_accuracy, "Epoch": epoch})
                self.logger.write(val_meter.return_msg() + f" [accuracy] {val_accuracy}", mode='val')

                if val_accuracy < self.metric_best:
                    self.metric_best = val_accuracy
                    self.save_model(epoch, val_accuracy)

        print()
        self.logger.record("Training complete! Generating test predictions...", mode='info') 





