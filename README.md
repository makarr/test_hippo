`test_hippo.ipynb` offers a brief introduction to using HiPPO for audio at different sample rates.  HiPPO is the work of Albert Gu, et al, at Stanford and I take no credit for his research.  This is simply a demonstration of its utility for those interested in neural networks for real-time audio applications.  Whereas recurrent models like GRU and LSTM need to be trained for particular sample rates, HiPPO-based models can accommodate arbitrary sample rates, as long as the training data is sufficiently diverse.  Also, whereas GRU and LSTM have 3 or 4 gates and use tanh activations, HiPPO is a single linear layer with no activation.  This means that, for real-time use, HiPPO-based models can be at least 3x or 4x faster.

HiPPO is foundational to a more powerful model called S4, which has the further advantage of being switchable between recurrent and convolutional modes, meaning fast parallel training and fast sequential inference -- the best of both worlds!  However, the demonstration here uses a bare HiPPO model that acts as a simple feature encoder projecting a 1-dimensional signal into N dimensions, similar to the "ih" layers in gated recurrent models.

Future experiments will explore S4 in all its glory. But in the meantime, please visit https://github.com/HazyResearch/state-spaces, which also contains the source code for `unroll.py` and `standalone.py` in this repository.

Required libraries:
numpy
torch
torchaudio
scipy
einops
