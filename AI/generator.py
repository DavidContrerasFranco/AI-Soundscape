import numpy as np
import torch
import sys
from soundscape import Soundscape
from trainer import use_model
from scipy.io import savemat
sys.path.append("..")

def main():
    # Load Model
    model = torch.load("../AI/ai_soundscape.pt")
    freqs = model.input_dimension

    # Set to eval
    model.eval()

    # Time lenght of eval and total duration
    seq_len = 60
    duration = 10

    # First state noise
    init_data_Noise = torch.randn(freqs, 2, seq_len).to(device)
    hidden_Re_Noise, hidden_Im_Noise = model.init_hidden_noise(seq_len=seq_len, device=device)

    # First state silence
    hidden_Re_Silence, hidden_Im_Silence = model.init_hidden_silence(seq_len=seq_len, device=device)
    init_data_Silence = torch.zeros(freqs, 2, seq_len).to(device)

    # Generate raw data
    print("-------------\nGenerator with initial Noise:\n")
    raw_data_Noise = use_model(init_data_Noise, model, hidden_Re_Noise, hidden_Im_Noise, duration, seq_len)
    print("-------------\nGenerator with initial Silence:\n")
    raw_data_Silence = use_model(init_data_Silence, model, hidden_Re_Silence, hidden_Im_Silence, duration, seq_len)

    # Saving into MAT files
    savemat('../DataGenerated/noise.mat', {'Answ':raw_data_Noise})
    savemat('../DataGenerated/silence.mat', {'Answ':raw_data_Silence})


if __name__ == '__main__':

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    
    main()