from datetime import datetime
from time import sleep
import os
import torch


def get_latest_model(prefix=""):
    files = [f for f in os.listdir("models") if f.endswith("pt") and f.startswith(prefix)]
    return files[0]


def save_checkpoint(model, name, frames=0, mean_reward=0, overwrite=False, loc="models/"):
    if overwrite:
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(loc, name + ".pt"))
        return
    now = datetime.now()
    hm = now.strftime("%H%M%S")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean_reward": mean_reward,
            "frames": frames,
        },
        os.path.join(loc, name + "_" + hm + ".pt"),
    )


def load_checkpoint(model, name, device):
    print("Load model", name)
    sleep(1)
    checkpoint = torch.load("models/" + name, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def get_cuda_device():
    """Get the pytorch cuda device if available.

    Returns:
        {torch.device} -- A cuda device if available, otherwise cpu
    """
    if torch.cuda.is_available():
        # cuda:0 will still use all GPUs
        device = torch.device("cuda:0")
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = torch.device("cpu")
    return device
