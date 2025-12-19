# Diffusion-sandbox
An experimental sandbox for implementing, testing, and visualizing diffusion models.



### 🔢 Digit diffusion

The figures below show the reverse diffusion process on MNIST digits:  
each row corresponds to a target digit (0–5), and columns show samples
evolving from pure noise (left) to a clean digit (right).

<p align="center">
  <img src="assets/digit_diffusion.png" alt="Digit generation with diffusion models" width="700">
</p>

<p align="center">
  <img src="assets/digit_diffusion2.png" alt="Digit generation with diffusion models2" width="700">
</p>

### 🎲 Dot → 🔢 digit diffusion 

This variant conditions the reverse diffusion process **only on a sparse “dot image”**:
an input canvas with *N* bright dots (single pixels), where *N* encodes the target class.
At inference time, I provide the dot image and the model denoises from pure noise into
a clean MNIST digit consistent with that dot-conditioning.

<img width="1599" height="126" alt="image" src="https://github.com/user-attachments/assets/ee81ab75-5e3a-414c-97d6-a8c07abec54b" />
<img width="1599" height="126" alt="image" src="https://github.com/user-attachments/assets/d965d19e-7981-43f1-bcb7-2798fba5410d" />
