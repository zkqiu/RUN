"""
Agentic RL（多轮工具调用 + GRPO）核心伪代码

任务：数学/编程题，模型可以多次写 Python 代码交给 sandbox 执行，拿到结果后继续推理，最后给出答案。

交互协议（模型输出格式）：
    <think> ... </think>
    <code>
    ```python
    print(1 + 1)
    ```
    </code>
    -> 环境追加: <output> 2 </output>
    ...
    <answer> 2 </answer>

组件：
    1. OutputParser   解析 LLM 输出：动作类型（code / answer / invalid）+ 内容
    2. CodeSandbox    隔离执行代码：子进程 + 超时 + 资源限制 + 输出截断
    3. AgentEnv       把 parser 和 sandbox 串成 step(action_text) -> (obs, done)
    4. rollout        多轮生成，维护 token 级 loss_mask（工具返回的 token 不算策略梯度）
    5. reward_fn      结果奖励 + 格式约束
    6. grpo_update    组内相对优势 + clipped surrogate + KL
"""
import os
import re
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ============================== 1. Parser ================================
@dataclass
class Action:
    kind: str            # "code" | "answer" | "invalid"
    content: str = ""
    error: str = ""


class OutputParser:
    CODE_RE = re.compile(r"<code>\s*(?:```(?:python|py)?\s*\n)?(.*?)(?:```\s*)?</code>", re.S)
    ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

    def parse(self, text: str) -> Action:
        # 以"最先出现"的动作为准：生成时 stop 在 </code> 或 </answer>，正常一轮只会有一个动作；
        # 若模型违规输出多个，只执行第一个，后面的内容视为噪声
        code_m = self.CODE_RE.search(text)
        ans_m = self.ANSWER_RE.search(text)
        if code_m and (not ans_m or code_m.start() < ans_m.start()):
            code = code_m.group(1).strip()
            if not code:
                return Action("invalid", error="empty code block")
            return Action("code", code)
        if ans_m:
            return Action("answer", ans_m.group(1).strip())
        return Action("invalid", error="no <code> or <answer> tag found")


# ============================== 2. Sandbox ===============================
@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timeout: bool = False

    def to_observation(self, max_chars: int = 1024) -> str:
        if self.timeout:
            body = "TimeoutError: execution exceeded time limit"
        elif self.exit_code != 0:
            # 只保留 traceback 末尾，最有用的信息在最后几行，且避免超长 obs 撑爆上下文
            body = (self.stdout + "\n" + self.stderr[-max_chars:]).strip()
        else:
            body = self.stdout.strip() or "(no output, use print() to show results)"
        if len(body) > max_chars:
            body = body[: max_chars // 2] + "\n...[truncated]...\n" + body[-max_chars // 2:]
        return f"\n<output>\n{body}\n</output>\n"


class CodeSandbox:
    """
    本地子进程版 sandbox。生产环境应换成 docker / gVisor / firecracker / e2b 等强隔离方案，
    接口保持 run(code) -> ExecResult 不变即可。
    """

    def __init__(self, timeout: float = 5.0, mem_mb: int = 512, max_output: int = 64 * 1024):
        self.timeout = timeout
        self.mem_mb = mem_mb
        self.max_output = max_output

    def _limit(self):
        # 在子进程 exec 前设置资源上限，防止模型写出死循环分配内存 / fork 炸弹 / 写大文件
        mem = self.mem_mb * 1024 * 1024
        for res, val in [(getattr(resource, "RLIMIT_AS", None), mem),
                         (resource.RLIMIT_CPU, int(self.timeout) + 1),
                         (resource.RLIMIT_FSIZE, 10 * 1024 * 1024),
                         (getattr(resource, "RLIMIT_NPROC", None), 64)]:
            if res is None:
                continue
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass  # macOS 不支持部分 rlimit，忽略即可

    def run(self, code: str) -> ExecResult:
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "main.py")
            with open(path, "w") as f:
                f.write(code)
            try:
                p = subprocess.run(
                    [sys.executable, "-I", path],  # -I: 隔离模式，不读用户 site-packages / 环境变量
                    cwd=workdir, capture_output=True, text=True, timeout=self.timeout,
                    preexec_fn=self._limit, env={"PATH": "/usr/bin:/bin"},
                )
                return ExecResult(p.stdout[: self.max_output], p.stderr[: self.max_output], p.returncode)
            except subprocess.TimeoutExpired as e:
                out = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
                return ExecResult(out[: self.max_output], "", -1, timeout=True)


# ============================== 3. Env ===================================
@dataclass
class Trajectory:
    prompt_ids: list
    response_ids: list = field(default_factory=list)
    loss_mask: list = field(default_factory=list)   # 1 = 模型生成的 token，0 = 环境注入的 token
    answer: str = None
    num_tool_calls: int = 0
    num_invalid: int = 0
    finished: bool = False


class AgentEnv:
    def __init__(self, sandbox: CodeSandbox, parser: OutputParser, max_turns: int = 8):
        self.sandbox = sandbox
        self.parser = parser
        self.max_turns = max_turns

    def step(self, action_text: str, traj: Trajectory):
        """返回 (observation_text or None, done)"""
        action = self.parser.parse(action_text)
        if action.kind == "answer":
            traj.answer = action.content
            return None, True
        if action.kind == "code":
            traj.num_tool_calls += 1
            return self.sandbox.run(action.content).to_observation(), False
        traj.num_invalid += 1
        # 给出格式错误反馈而不是直接终止，让模型有机会自我纠正
        return f"\n<output>\nFormatError: {action.error}\n</output>\n", False


# ============================== 4. Rollout ===============================
SYSTEM_PROMPT = (
    "Solve the problem step by step. You may run Python by writing\n"
    "<code>\n```python\n...\n```\n</code>\n"
    "and you will get the result in <output></output>. "
    "Put the final answer in <answer></answer>."
)
STOP = ["</code>", "</answer>"]


def rollout_batch(llm, tokenizer, env: AgentEnv, problems, group_size, cfg):
    """
    对每道题采样 group_size 条轨迹。所有活跃轨迹每轮一起 batch generate（vLLM / SGLang），
    sandbox 执行用线程池并发，避免被慢代码拖住整批 GPU。
    """
    trajs, metas = [], []
    for prob in problems:
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prob["question"]}],
            add_generation_prompt=True,
        )
        for _ in range(group_size):
            trajs.append(Trajectory(prompt_ids=list(ids)))
            metas.append(prob)

    pool = ThreadPoolExecutor(max_workers=cfg.sandbox_workers)
    for turn in range(env.max_turns):
        active = [t for t in trajs if not t.finished]
        if not active:
            break
        # 上下文 = prompt + 已有 response（含之前的工具输出）
        outs = llm.generate(
            [t.prompt_ids + t.response_ids for t in active],
            sampling_params=dict(temperature=1.0, max_tokens=cfg.max_tokens_per_turn,
                                 stop=STOP, include_stop_str_in_output=True),
        )
        for t, out in zip(active, outs):
            t.response_ids += out.token_ids
            t.loss_mask += [1] * len(out.token_ids)

        texts = [tokenizer.decode(out.token_ids) for out in outs]
        results = list(pool.map(lambda args: env.step(*args), zip(texts, active)))

        for t, (obs, done) in zip(active, results):
            if done:
                t.finished = True
                continue
            obs_ids = tokenizer.encode(obs, add_special_tokens=False)
            t.response_ids += obs_ids
            t.loss_mask += [0] * len(obs_ids)   # 工具输出不是策略的动作，不计算梯度
            if len(t.prompt_ids) + len(t.response_ids) >= cfg.max_total_len:
                t.finished = True               # 超长截断，answer 仍为 None
    pool.shutdown()
    return trajs, metas


# ============================== 5. Reward ================================
def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "").rstrip(".").lower()


def reward_fn(traj: Trajectory, meta: dict) -> float:
    # 以结果奖励为主，避免给中间过程设计复杂 reward 导致 hacking（如刷工具调用次数）
    if traj.answer is None:
        return -1.0                         # 没给答案 / 超轮数 / 超长
    r = 1.0 if normalize(traj.answer) == normalize(meta["answer"]) else 0.0
    r -= 0.1 * traj.num_invalid             # 轻微惩罚格式错误
    return r


# ============================== 6. GRPO ==================================
def pad_batch(trajs, pad_id, device):
    seqs = [t.prompt_ids + t.response_ids for t in trajs]
    L = max(map(len, seqs))
    input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
    for i, (t, s) in enumerate(zip(trajs, seqs)):
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
        loss_mask[i, len(t.prompt_ids): len(s)] = torch.tensor(t.loss_mask, dtype=torch.float)
    # 与 logits[:, :-1] 对齐：位置 t 的 logp 预测的是 token t+1
    return input_ids.to(device), attn.to(device), loss_mask[:, 1:].to(device)


def token_logprobs(model, input_ids, attn):
    logits = model(input_ids, attention_mask=attn).logits[:, :-1].float()
    return torch.gather(F.log_softmax(logits, -1), -1, input_ids[:, 1:, None]).squeeze(-1)


def grpo_advantages(rewards: torch.Tensor, group_size: int, eps=1e-6):
    # 同一题的 G 条轨迹互为 baseline，省掉 critic
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
    return adv.view(-1)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, clip_eps=0.2, kl_coef=0.01):
    ratio = torch.exp(logp - old_logp)
    adv = adv[:, None]
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    # k3 估计器：无偏且恒非负
    kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
    per_tok = pg + kl_coef * kl
    # token-level 平均（DAPO 做法），长轨迹不会被按序列平均稀释
    return (per_tok * mask).sum() / mask.sum().clamp(min=1)


def train(policy, ref, llm, tokenizer, dataloader, cfg):
    sandbox = CodeSandbox(timeout=cfg.exec_timeout)
    env = AgentEnv(sandbox, OutputParser(), max_turns=cfg.max_turns)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    for step, problems in enumerate(dataloader):
        # ---- rollout（推理引擎，权重需与 policy 同步）----
        trajs, metas = rollout_batch(llm, tokenizer, env, problems, cfg.group_size, cfg)
        rewards = torch.tensor([reward_fn(t, m) for t, m in zip(trajs, metas)])
        adv = grpo_advantages(rewards, cfg.group_size)

        # 组内奖励全相同 => 优势全 0，无梯度信号，过滤掉节省算力（DAPO dynamic sampling）
        keep = rewards.view(-1, cfg.group_size).std(1) > 0
        keep = keep.repeat_interleave(cfg.group_size)
        trajs = [t for t, k in zip(trajs, keep) if k]
        adv = adv[keep].to(cfg.device)
        if not trajs:
            continue

        input_ids, attn, mask = pad_batch(trajs, tokenizer.pad_token_id, cfg.device)
        with torch.no_grad():
            old_logp = token_logprobs(policy, input_ids, attn)
            ref_logp = token_logprobs(ref, input_ids, attn)

        # ---- update ----
        for _ in range(cfg.ppo_epochs):
            for idx in torch.randperm(len(trajs)).split(cfg.mini_batch_size):
                logp = token_logprobs(policy, input_ids[idx], attn[idx])
                loss = grpo_loss(logp, old_logp[idx], ref_logp[idx], adv[idx], mask[idx],
                                 cfg.clip_eps, cfg.kl_coef)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()

        llm.load_weights(policy.state_dict())  # 同步到推理引擎，保证下轮 on-policy
        print(dict(step=step, reward=rewards.mean().item(),
                   acc=(rewards > 0.5).float().mean().item(),
                   tool_calls=sum(t.num_tool_calls for t in trajs) / len(trajs)))


# ============================== Demo =====================================
if __name__ == "__main__":
    # 只验证 parser + sandbox + env 这部分可运行逻辑，不涉及模型
    env = AgentEnv(CodeSandbox(timeout=2), OutputParser())
    traj = Trajectory(prompt_ids=[])
    cases = [
        "<think>算一下</think>\n<code>\n```python\nprint(sum(range(101)))\n```\n</code>",
        "<code>\n```python\n1/0\n```\n</code>",
        "<code>\n```python\nwhile True: pass\n```\n</code>",
        "我觉得答案是 5050",
        "<answer>5050</answer>",
    ]
    for c in cases:
        obs, done = env.step(c, traj)
        print(f"--- action:\n{c}\n--- obs: {obs!r}  done={done}\n")
    print("answer:", traj.answer, "reward:", reward_fn(traj, {"answer": "5050"}))
