# AnoGAN

Original paper by: [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921)

<img src= './images/anomally.png' width=100% height=15%/>

</br>

Tested fot detecting anomality on a single cup.

|<center>cup normal|<center>cup contaminated|
|--|--|
|<img src="./images/nomal_cup.jpeg" width=150 height=150>|<img src="./images/dirt_cup.jpeg" width=150 height=150>|

## Training with DCGAN

paper: [DCGAN](/home/junshick/Workspace/Study/GAN/DCGAN/images/nomal_cup.jpeg)
- First, this model is trained with DCGAN
- Anomally detection is executed at inference session.
- Image size is resized to 64x64 and applied general convolution generator and discriminator
- Trained with 67 normal cup images.
<center>
    <img src="./images/ezgif.com-gif-maker.gif" width= 80%>
</center>
</br>

## Inference (Anomally Detection)

- Train latent vector so that D(G(z)) looks as much simillar as input x.
- Show difference between generated fake image and real image with anomally score.

    |<center>real image|<center>generated image|<center>difference|
    |--|--|--|
    |<img src="./images/inf/inf_real2.png" width=120 >|<img src="./images/inf/inf_gen2.png" width=120>|<img src="./images/inf/inf_diff2.png" width=120>|
    |<img src="./images/inf/inf_real1.png" width=120 >|<img src="./images/inf/inf_gen1.png" width=120>|<img src="./images/inf/inf_diff1.png" width=120>|
    |<img src="./images/inf/inf_real3.png" width=120 >|<img src="./images/inf/inf_gen3.png" width=120>|<img src="./images/inf/inf_diff3.png" width=120>|










