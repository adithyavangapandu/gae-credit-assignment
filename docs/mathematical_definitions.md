# Mathematical definitions

Time t indexes transitions 0,...,T-1. Reward r_t is received after action a_t
produces state s_(t+1); d_t is true exactly on a terminal transition. Full
episodes end at T=200. Time remaining is observed and V(s_T) contributes zero.

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\qquad
q=\gamma\lambda,\quad\gamma=0.995,\quad\lambda=0.95.
\]

Let L_t count residuals through the first terminal transition (or the end of the
provided array). Finite GAE is

\[
\hat A_t^{(H)}=\sum_{k=0}^{\min(H,L_t)-1}q^k\delta_{t+k},\qquad
\hat V_t^{(H)}=V(s_t)+\hat A_t^{(H)}.
\]

H=1 is the TD(0) advantage; H=3 is the short estimator; H=16 is intermediate.
Full includes L_t residuals and uses reverse recursion
`A[t] = delta[t] + q*(1-terminated[t])*A[t+1]`.
Finite estimators require an actual cutoff. An unbounded reverse recursion is
full GAE, even if incorrectly labeled three-step. Both estimators use actual
rewards and frozen rollout critic predictions, not model-generated future rewards.
The terminal mask suppresses bootstrap values and prevents crossing resets.
For a nonterminal array boundary only, the final supplied value is bootstrapped;
the study collector always supplies complete truly terminal episodes.

The effective horizon is min(H,L_t) for finite H and L_t for full. Lambda=1 and
full horizon telescope to the finite discounted Monte Carlo return minus V(s_t).
Targets use raw advantages; normalization is applied only to the PPO actor loss.

## Hand-worked cutoff example

\[
A_9^{(3)}=\delta_9+q\delta_{10}+q^2\delta_{11},
\]
\[
A_9^{\mathrm{full}}=\delta_9+q\delta_{10}+q^2\delta_{11}
 +q^3\delta_{12}+\cdots.
\]

For the separate arithmetic example q=1/2, residuals [1,2,3,4], and a terminal
fourth transition: A^(1)=[1,2,3,4], A^(3)=[2.75,4.5,5,4], and
A^(full)=[3.25,4.5,5,4]. The omitted tail at the first transition is 0.5.
The effective H=3 horizons are [3,3,2,1]. With rewards [1,2,10,20], values
[3,4,5,6,999], gamma=1/2, lambda=1, and terminal masks [0,1,0,1], residuals
are [0,-2,8,14] and full advantages [-1,-2,15,14]; the 999 terminal value and
the next episode's residuals do not contribute across the boundary.

## Reward delivery

Let wrap map angles to [-pi,pi). The base reward is

\[
r_t^{dense}=-[\operatorname{wrap}(\theta_{t+1})^2
 +0.1\dot\theta_{t+1}^2+0.001u_t^2].
\]

For a delay block b,...,e, emit only at e:

\[
r_e^{batch}=\sum_{i=b}^e\gamma^{i-e}r_i^{dense}.
\]

All other block rewards are zero. Notice the **negative** exponents. Thus
`pending_corrected = pending_corrected/gamma + dense_reward` accumulates the
payout. Multiplying by gamma^e gives sum_i gamma^i*r_i, so full and partial
blocks preserve the episode's discounted return exactly in real arithmetic.
Floating-point tests use a tolerance. For gamma=1/2 and rewards [1,2] delayed
to the second transition, payout=4 and discounted return=2, matching 1+0.5*2.
Delays require gamma>0; D=1 equals dense. D=8 and D=32 are study conditions.

Sparse reward is 1 when abs(wrap(theta_(t+1)))<0.262 and
abs(angular_velocity_(t+1))<1, otherwise 0. Strict inequalities apply. Reward
repeats on every qualifying step. Success is a separate ten-step streak metric.

## Estimator diagnostics

Omitted tail: `full - truncated`, a signed quantity. Bounded tail fraction:
`abs(tail)/(abs(truncated)+abs(tail)+1e-12)`. This measures omitted magnitude
relative to retained plus omitted magnitude; it is not a percentage of full
advantage, which can be zero due to cancellation. Opposite strictly nonzero
signs count as sign disagreement; zero versus nonzero does not.

Direct event credit recomputes GAE with reward r_e set to zero, leaving all
critic predictions fixed, and subtracts that result from the original. It is
`r_e*q^(e-t)` if e lies within the estimator horizon and same episode, otherwise
zero. The often-used `q^distance` formula assumes a unit reward. For q=1/2,
a unit reward at index 3 gives full credit [1/8,1/4,1/2,1,0] and H=3 credit
[0,1/4,1/2,1,0]. Segment credit zeros rewards in the half-open interval
[start,stop) and equals their summed direct event credits. These are fixed-critic
algebraic diagnostics, not estimates of causal policy-performance effects.

## PPO density and objective

For latent z sampled from N(mu,sigma), action a=M*tanh(z):

\[
\log\pi(a|s)=\log\mathcal N(z;\mu,\sigma)
 -\log M-\log(1-\tanh^2z).
\]

Use the stable identity `log(1-tanh(z)^2)=2*(log(2)-z-softplus(-2*z))` and
store z at collection. Inverting a numerically saturated action loses z.
PPO ratios use the fixed old log probabilities. The actor minimizes the negative
minimum of unclipped and clipped ratio-weighted normalized advantage, minus the
entropy coefficient times transformed-policy Monte Carlo entropy. The critic
minimizes mean squared error to fixed value targets times the value coefficient.
The approximate KL diagnostic is mean[(ratio-1)-log(ratio)] on the rollout,
and the clip fraction is the fraction with abs(ratio-1)>epsilon.

## References

- [GAE paper](https://arxiv.org/abs/1506.02438).
- [PPO paper](https://arxiv.org/abs/1707.06347).
- [Gymnasium Pendulum conventions](https://gymnasium.farama.org/environments/classic_control/pendulum/).
- [uv locked environments and Docker](https://docs.astral.sh/uv/guides/integration/docker/).
