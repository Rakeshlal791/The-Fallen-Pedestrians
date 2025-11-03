# The-Fallen-Pedestrians
An anomaly detection Variational Autoencoder (VAE) model to identify unusual pedestrians in the scene.

# Description
Identifying anomalous pedestrian behavior is a crucial task for autonomous driving, robotic navigation, and pedestrian safety in general.

In this work, we define pedestrians with unusual poses—such as those in fallen or lying-down positions, which deviate from normal upright walking or standing poses—as anomalous in traffic scenarios.

# Dataset
We trained the VAE model on the [European City Person (ECP)](https://https://eurocity-dataset.tudelft.nl/) dataset to learn the latent distribution of normal pedestrians.

## Pedestrian Samples
Our dataset class returns a per-instance bounding box and normalized pose keypoints for training.

The following is how samples from the dataset appear
![alt text](https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/ped_samples.png?raw=true)

# Model and Training
A Variational Autoencoder (VAE) is employed to model pedestrian pose and bounding box features, capturing structural information such as bone angles and limb lengths.

![alt text](https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/vae_model.png?raw=true | width=100)

The input feature vector **x** encodes:

$$
\begin{aligned}
&\text{• 14 body bones} \times 2~(\cos\theta, \sin\theta) \Rightarrow 28~\text{values} \\
&\text{• 14 bone lengths (normalized, log-scaled)} \\
&\text{• Bounding box width and height (log-scaled and normalized)}
\end{aligned}
$$

Thus, the total feature dimension is:

$$
\mathbf{x} \in \mathbb{R}^{28 + 14 + 2 = 44}
$$

The encoder network models the latent distribution as

$$
q_{\phi}(z|x),
$$

which approximates a posterior Gaussian for each input sample.  
A standard normal prior,

$$
p(z) = \mathcal{N}(0, I),
$$

is assumed over the latent space, encouraging all encoded representations to cluster around a single global Gaussian distribution.

The decoder network,

$$
p_{\theta}(x|z),
$$

learns to reconstruct the original input feature vector from the latent code, thereby capturing the underlying structure of normal pedestrian poses.  
During training, the VAE minimizes the combined reconstruction and Kullback–Leibler (KL) divergence losses:

$$
\mathcal{L} = \mathbb{E}_{q_{\phi}(z|x)}[\lVert x - \hat{x}\rVert_1] + \beta D_{KL}(q_{\phi}(z|x)\lVert p(z)).
$$

After training on normal pedestrian poses, anomalous poses (e.g., fallen or lying pedestrians) are identified as samples with **high reconstruction error** or **low likelihood** under the learned latent distribution.

