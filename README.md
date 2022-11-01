# vicsek-swarm-control

We take the original Vicsek model described in (?):
$$x_i(t + \Delta t) = x_i(t) + v_i \Delta t$$
and we add a potential $\Psi$:

$$X_i(t+\Delta t) = X_i(t) + v_i \Delta t -  \nabla \psi (X_i(t+\Delta t), t) \Delta t$$

where $x_i$ is the position of the ith particle and its velocity $v_i$ is of modulus 1 and makes an angle $\theta$ s.t. 
$$\theta_i(t + \Delta t) = <\theta>_r + \eta_i$$
wehere $<\theta>_r$ is the average orientation of particles within radius $r$ of particle $i$ and $\eta_i$ is a random perturbation

![vicsek with potential](https://github.com/zakaryael/vicsek-swarm-control/blob/main/animation.gif)

<figure markdown>
<figcaption>Vicsek particles with a constant gaussian potential</figcaption>
</figure>

The aim is to control the potential's location to get the particles to perform certain tasks such as target searching or tracking etc.
