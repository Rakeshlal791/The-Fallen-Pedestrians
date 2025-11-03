# The-Fallen-Pedestrians
An anomaly detection Variational Autoencoder (VAE) model to identify unusual pedestrians in the scene.

# Description
Identifying anomalous pedestrian behavior is a crucial task for autonomous driving, robotic navigation, and pedestrian safety in general.

In this work, we define pedestrians with unusual poses—such as those in fallen or lying-down positions, which deviate from normal upright walking or standing poses—as anomalous in traffic scenarios.

# Dataset
We trained the VAE model on the [European City Person (ECP)](https://https://eurocity-dataset.tudelft.nl/) dataset to learn the latent distribution of normal pedestrians.

## Pedestrian Samples
Our dataset class returns a per-instance bounding box and normalized pose keypoints for training.

The following are some samples from the dataset.

<img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/ped_samples.png?raw=true" width="520">

# Model and Training
A Variational Autoencoder (VAE) is employed to model pedestrian pose and bounding box features, capturing structural information such as bone angles and limb lengths.

![alt text](https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/vae_model.png?raw=true)

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

# Test Data Generation
We utilized a small subset of the validation split, excluded from training, to obtain representative samples of normal pedestrian behavior. To evaluate anomaly detection performance, we generated synthetic images representing five atypical pedestrian postures: sitting, lying, crawling, bending/leaning, and imbalanced positions.

## Synthetic Data
We generated synthetic anomalous scenarios using [Gemini 2.5 Flash Image (Nano Banana)](https://aistudio.google.com/models/gemini-2-5-flash-image) by performing inpainting to introduce fallen pedestrians into selected images from the test dataset. Any visual artifacts produced during synthesis were manually corrected using the GNU Image Manipulation Program (GIMP). Finally, we annotated the pose keypoints and bounding boxes for each anomalous pedestrian.

We generated approximately 50 synthetic test images per category, annotated, and applied small Gaussian perturbations to pose keypoints, introducing subtle variations that better capture real-world posture diversity.

Below are some examples of our synthetic pedestrians for each anomalous category.

| Category | Examples |
|-----------|-----------|
| **Sitting** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/sitting_row.png?raw=true" width="520"> |
| **Bending** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/bending_row.png?raw=true" width="520"> |
| **Crawling** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/crawling_row.png?raw=true" width="520"> |
| **Lying** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/lying_row.png?raw=true" width="520"> |
| **Imbalanced** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/imbalanced_row.png?raw=true" width="520"> |

---

## Results

To evaluate our model, we used approximately **1.2k non-anomalous** and **1.2k anomalous** pedestrian samples.  
We visualize the learned latent representations ($$\mu$$) using **Principal Component Analysis (PCA)**, comparing the distributions of normal and anomalous data in the latent space.

<img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/main/images/latent_mu.png?raw=true" width="520">


