import sys
import torch


if __name__ == "__main__":
    state = torch.load(sys.argv[1], map_location="cpu")["model"]

    for key in state:
        print(key)
