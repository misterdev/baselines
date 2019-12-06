# from ray.rllib.policy.tf_policy_template import build_tf_policy
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer

ray.init()
trainer = PPOTrainer(env="CartPole-v0", config={"train_batch_size": 4000})
while True:
    print(trainer.train())

# from ray.rllib.agents.ppo.ppo_policy import ppo_surrogate_loss, \
#     kl_and_loss_stats, \
#     vf_preds_and_logits_fetches, \
#     postprocess_ppo_gae, \
#     clip_gradients, \
#     setup_mixins, \
#     LearningRateSchedule, KLCoeffMixin, ValueNetworkMixin

# from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
# from ray.rllib.agents.trainer_template import build_trainer

# PPOTFPolicy = build_tf_policy(
#     name="PPOTFPolicy",
#     get_default_config=lambda: DEFAULT_CONFIG,
#     loss_fn=ppo_surrogate_loss,
#     stats_fn=kl_and_loss_stats,
#     extra_action_fetches_fn=vf_preds_and_logits_fetches,
#     postprocess_fn=postprocess_ppo_gae,
#     gradients_fn=clip_gradients,
#     before_loss_init=setup_mixins,
#     mixins=[LearningRateSchedule, KLCoeffMixin, ValueNetworkMixin])

# PPOTrainer = build_trainer(
#     name="PPOTrainer",
#     default_config=DEFAULT_CONFIG,
#     default_policy=PPOTFPolicy,
#     make_policy_optimizer=choose_policy_optimizer,
#     validate_config=validate_config,
#     after_optimizer_step=update_kl,
#     before_train_step=warn_about_obs_filter,
#     after_train_result=warn_about_bad_reward_scales)