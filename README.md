# AI Soundscape

This is the source code of the AI Soundscape project. Which tries to create a sound that recreates an environment. It does this by spectral analysis and a neural network that uses such components as training data. 

# Files

This project is divided into data processing and the neural network.

## AI

In this folder everything regarding the neural network is located:

- soundscape.py: Main file that defines the structure of the neural network and trains it using the trainer.
- trainer.py: Trainer file that contains the single epoch run function, the training procedure and a function to generate new data with a trained model.
- generator.py: File that creates data from silence and from noise using a trained model.
- utils.py: Helper functions.


## DataGenerated

This folder contains raw data in .mat files that were generated using the neural network.

## DataProcess

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## SoundGenerator

This folder contains a .m file that generated audio from raw data.