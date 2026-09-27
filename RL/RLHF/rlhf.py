"""
RLHF (InstructGPT 风格) 训练伪代码：SFT -> Reward Model -> PPO

四个模型：
    actor   : 策略模型，由 SFT 初始化，被训练
    ref     : SFT 模型冻结副本，用于 KL 约束，防止 actor 偏离太远导致 reward hacking
    reward  : 奖励模型，冻结，对完整回复打一个标量分
    critic  : 价值模型，通常由 RM 初始化，被训练，逐 token 估计 V(s_t)
"""
import torch
import torch.nn.functional as F


# ============================== Stage 1: SFT ==============================
def sft_step(model, batch, optimizer):
    # 只在 response 部分计算 loss，prompt 部分 label=-100，避免模型去学习复述 prompt
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()


# ========================= Stage 2: Reward Model ==========================
def rm_step(reward_model, batch, optimizer):
    # Bradley-Terry: P(chosen > rejected) = sigmoid(r_c - r_r)
    # reward_model 输出取最后一个非 pad token 的标量，作为整段回复的分数
    r_chosen = reward_model(batch["chosen_ids"], batch["chosen_mask"])       # [B]
    r_rejected = reward_model(batch["rejected_ids"], batch["rejected_mask"]) # [B]
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item(), (r_chosen > r_rejected).float().mean().item()


# ============================= Stage 3: PPO ===============================
def token_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    # 从后往前递推：delta_t = r_t + gamma*V_{t+1} - V_t ; A_t = delta_t + gamma*lam*A_{t+1}
    T = rewards.size(1)
    adv = torch.zeros_like(rewards)
    last = 0
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else 0.0
        delta = rewards[:, t] + gamma * next_v * mask[:, t] - values[:, t]
        last = delta + gamma * lam * last * mask[:, t]
        adv[:, t] = last
    returns = adv + values
    return adv, returns


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x, mask, eps=1e-8):
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps)


@torch.no_grad()
def make_experience(actor, ref, reward_model, critic, tokenizer, prompts, cfg):
    # 1. rollout
    seqs = actor.generate(prompts, max_new_tokens=cfg.max_new_tokens, do_sample=True)
    attn = (seqs != tokenizer.pad_token_id).long()
    # action_mask 只覆盖 response token，prompt 不参与策略梯度
    action_mask = build_response_mask(seqs, prompt_len=prompts.size(1), attn=attn)[:, 1:]

    # 2. 四个模型各跑一次前向
    old_logp = token_logprobs(actor, seqs, attn)
    ref_logp = token_logprobs(ref, seqs, attn)
    values = critic(seqs, attn)[:, :-1]            # [B, T-1]
    score = reward_model(seqs, attn)               # [B]

    # 3. 逐 token 奖励：每个 token 都有 -beta*KL，RM 分数只加在最后一个 response token 上
    #    KL 做成 reward 的一部分（而非 loss 项），这样 critic 能把它一并纳入价值估计
    kl = old_logp - ref_logp
    rewards = -cfg.kl_coef * kl
    last_idx = action_mask.sum(1).long() + prompts.size(1) - 2
    rewards[torch.arange(len(seqs)), last_idx] += score.clamp(-cfg.reward_clip, cfg.reward_clip)
    rewards = rewards * action_mask

    # 4. GAE
    adv, returns = compute_gae(rewards, values, action_mask, cfg.gamma, cfg.lam)
    adv = masked_whiten(adv, action_mask)

    return dict(seqs=seqs, attn=attn, action_mask=action_mask, old_logp=old_logp,
                values=values, adv=adv, returns=returns, score=score, kl=kl)


def ppo_update(actor, critic, actor_opt, critic_opt, exp, cfg):
    mask = exp["action_mask"]

    # ---- actor: clipped surrogate objective ----
    logp = token_logprobs(actor, exp["seqs"], exp["attn"])
    ratio = torch.exp(logp - exp["old_logp"])
    pg1 = -exp["adv"] * ratio
    pg2 = -exp["adv"] * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
    actor_loss = masked_mean(torch.max(pg1, pg2), mask)
    actor_opt.zero_grad(); actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
    actor_opt.step()

    # ---- critic: clipped value loss，防止 value 一步跳太远 ----
    v = critic(exp["seqs"], exp["attn"])[:, :-1]
    v_clip = exp["values"] + (v - exp["values"]).clamp(-cfg.value_clip, cfg.value_clip)
    vf_loss = 0.5 * masked_mean(
        torch.max((v - exp["returns"]) ** 2, (v_clip - exp["returns"]) ** 2), mask
    )
    critic_opt.zero_grad(); vf_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
    critic_opt.step()

    return actor_loss.item(), vf_loss.item()


def rlhf_ppo_train(actor, ref, reward_model, critic, tokenizer, dataloader, cfg):
    ref.eval(); reward_model.eval()
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)

    for step, prompts in enumerate(dataloader):
        exp = make_experience(actor, ref, reward_model, critic, tokenizer, prompts, cfg)
        # 同一批经验复用 ppo_epochs 次，靠 clip 保证不会离 old policy 太远
        for _ in range(cfg.ppo_epochs):
            for mb in split_minibatch(exp, cfg.mini_batch_size):
                a_loss, v_loss = ppo_update(actor, critic, actor_opt, critic_opt, mb, cfg)
        log(step=step, reward=exp["score"].mean(), kl=masked_mean(exp["kl"], exp["action_mask"]),
            actor_loss=a_loss, critic_loss=v_loss)


# ---- 伪代码辅助函数 ----
def build_response_mask(seqs, prompt_len, attn):
    mask = attn.clone()
    mask[:, :prompt_len] = 0
    return mask


def split_minibatch(exp, size):
    B = exp["seqs"].size(0)
    idx = torch.randperm(B)
    for i in range(0, B, size):
        j = idx[i:i + size]
        yield {k: (v[j] if torch.is_tensor(v) else v) for k, v in exp.items()}


def log(**kw):
    print({k: (v.item() if torch.is_tensor(v) else v) for k, v in kw.items()})
