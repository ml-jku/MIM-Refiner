import math
from argparse import ArgumentParser
from pathlib import Path

import einops
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, Normalize, InterpolationMode
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="mae_refined_l16")
    parser.add_argument("--data_train", type=str, required=True)
    parser.add_argument("--data_test", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--testrun", action="store_true")
    return vars(parser.parse_args())


@torch.no_grad()
def main(model, data_train, data_test, device, accelerator, num_workers, batch_size, k, tau, testrun):
    # init model
    if testrun:
        print("testrun -> using testmodel")
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
    else:
        print(f"loading model '{model}'")
        model = torch.hub.load("ml-jku/MIM-Refiner", model).eval()

    # init transform
    transform = Compose([
        Resize(size=256, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size=224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # init datasets
    data_train = Path(data_train).expanduser()
    print(f"initializing train dataset '{data_train.as_posix()}'")
    dataset_train = ImageFolder(root=data_train, transform=transform)
    data_test = Path(data_test).expanduser()
    print(f"initializing test dataset '{data_test.as_posix()}'")
    dataset_test = ImageFolder(root=data_test, transform=transform)
    if testrun:
        print("testrun -> limit dataset size to 10")
        dataset_train = Subset(dataset_train, indices=list(range(10)))
        dataset_test = Subset(dataset_test, indices=list(range(10)))

    # initialize device
    if accelerator == "cpu":
        device = torch.device("cpu")
    elif accelerator == "gpu":
        device = torch.device(f"cuda:{device}")
    else:
        raise NotImplementedError(f"invalid accelerator '{accelerator}'")
    print(f"initialized device: '{device}'")
    model = model.to(device)

    # dont use multi-processing dataloading for testrun
    if testrun:
        num_workers = 0

    # extract train features and labels
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_x = []
    train_y = []
    for x, y in tqdm(dataloader_train):
        x = x.to(device)
        x = model(x)
        train_x.append(x.cpu())
        train_y.append(y.clone())
    train_x = torch.concat(train_x)
    train_y = torch.concat(train_y)

    # extract test features and labels
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_x = []
    test_y = []
    for x, y in tqdm(dataloader_test):
        x = x.to(device)
        x = model(x)
        test_x.append(x.cpu())
        test_y.append(y.clone())
    test_x = torch.concat(test_x)
    test_y = torch.concat(test_y)

    # knn
    accuracy = knn(
        train_x=train_x.to(device),
        train_y=train_y.to(device),
        test_x=test_x.to(device),
        test_y=test_y.to(device),
        k=k,
        tau=tau,
        batch_size=batch_size,
    )
    print(f"k-NN accuracy: {accuracy}")


def knn(train_x, train_y, test_x, test_y, k=10, tau=0.07, batch_size=128, eps=1e-6):
    # normalize to mean=0 std=1
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True) + eps
    train_x.sub_(mean).div_(std)
    test_x.sub_(mean).div_(std)

    # normalize to length 1
    train_x.div_(train_x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))

    # initialize onehot vector per class (used for counting votes in classification)
    num_classes = max(train_y.max(), test_y.max()) + 1
    print(f"number of classes: {num_classes}")
    class_onehot = torch.diag(torch.ones(max(2, num_classes), device=train_x.device))

    # predict in chunks to avoid OOM
    num_correct = 0
    num_chunks = math.ceil(len(test_y) / (batch_size or len(test_y)))
    for test_x_chunk, test_y_chunk in zip(test_x.chunk(num_chunks), test_y.chunk(num_chunks)):
        # retrieve the k NNs and their labels
        similarities = test_x_chunk @ train_x.T
        topk_similarities, topk_indices = similarities.topk(k=k, dim=1)
        flat_topk_indices = einops.rearrange(topk_indices, "num_test knn -> (num_test knn)")
        flat_nn_labels = train_y[flat_topk_indices]

        # calculate accuracy of a knn classifier
        flat_nn_onehot = class_onehot[flat_nn_labels]
        nn_onehot = einops.rearrange(
            flat_nn_onehot,
            "(num_test k) num_classes -> k num_test num_classes",
            k=k,
        )
        topk_similarities.div_(tau).exp_()
        topk_similarities = einops.rearrange(topk_similarities, "num_test knn -> knn num_test 1")
        logits = (nn_onehot * topk_similarities).sum(dim=0)
        y_hat_chunk = logits.argmax(dim=1)
        num_correct += (test_y_chunk == y_hat_chunk).sum()
    accuracy = num_correct / len(test_x)

    return accuracy


if __name__ == "__main__":
    main(**parse_args())
