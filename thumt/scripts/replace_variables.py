import re
import sys
import torch


def replace(vars1, vars2):
    for key in vars2:
        if key in vars1:
            sys.stdout.write("copy variable %s\n" % key)
            vars1[key].copy_(vars2[key])


def replace2(vars1, vars2):
    for key in vars2:
        if key == "source_embedding":
            key1 = "target_embedding"
            sys.stdout.write("copy variable %s\n" % key1)
            vars1[key1].copy_(vars2[key])
        elif key == "bias":
            key1 = "autoenc_bias"
            sys.stdout.write("copy variable %s\n" % key1)
            vars1[key1].copy_(vars2[key])
        else:
            key1 = key.replace("encoder", "autoenc")

            if key1 in vars1:
                sys.stdout.write("copy variable %s\n" % key1)
                vars1[key1].copy_(vars2[key])
                continue


if __name__ == "__main__":
    states_1 = torch.load(sys.argv[1], map_location="cpu")
    states_2 = torch.load(sys.argv[2], map_location="cpu")
    states_3 = torch.load(sys.argv[3], map_location="cpu")

    replace(states_1["model"], states_2["model"])
    replace2(states_1["model"], states_3["model"])

    torch.save(states_1, sys.argv[1])
