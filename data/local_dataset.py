import torch

raw_src = torch.tensor([
    [5, 6, 4, 4.5, 5, 8, 9, 8.25, 7, 8.2, 7.9, 8.5, 9, 10]
]).transpose(0, 1)

raw_tgt = torch.tensor([
    [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
]).transpose(0, 1)


def main():
    print(raw_src.shape)
    print(raw_tgt.shape)


if __name__ == '__main__':
    main()