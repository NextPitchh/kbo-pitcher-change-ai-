"""
KBO WPA Pipeline — DQN 에이전트 학습

행동: 현재 투수 1명 + 불펜 3명 = 총 4명 중 1명 선택
불펜 3명 선택 기준:
  1번 - pitcher_vs_batter_avg 가장 낮은 투수 (타자 상성 최우선)
  2번 - total_pitch_count 가장 낮은 투수 (가장 신선한 투수)
  3번 - 위 두 조건 종합 점수 중간값 (균형형)

보상: U = -E[WPA] - λ·Var[WPA]  (수비팀 관점 마코위츠 효용)
저장: dqn_output/dqn_agent_lambda{λ}.pth

실행: python dqn_agent.py
요구사항: pip install torch catboost pandas numpy pyarrow
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from monte_carlo_simulate import MonteCarloSimulator, GameState

# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
OUTPUT_DIR = Path("dqn_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {DEVICE}")

# ──────────────────────────────────────────────
# 1. 불펜 선택 로직
#    현재 투수 제외 전체 풀에서 3명 추출
#    → 현재 투수 포함 총 4명이 DQN 행동 공간
# ──────────────────────────────────────────────
def select_bullpen(
    pitcher_pool: list[dict],
    current_pitcher_id: str,
    batter_hit_type: str = "R",
    max_pitch_count: int = 120,   # 피로도 하드컷
) -> list[dict]:
    """
    전체 투수 풀에서 현재 투수를 제외하고
    3명의 불펜 후보를 선택.

    선택 기준:
      1번 — pitcher_vs_batter_avg 최소 (타자 상성)
      2번 — total_pitch_count 최소 (신선도)
      3번 — 두 조건 종합 점수 중간값 (균형형)
    """
    # 피로도 하드컷 + 현재 투수 제외
    pool = [
        p for p in pitcher_pool
        if p["pitcher_id"] != current_pitcher_id
        and p.get("total_pitch_count", 0) < max_pitch_count
    ]

    if len(pool) == 0:
        return []
    if len(pool) <= 3:
        return pool

    # 각 기준별 정규화 점수 (0~1, 낮을수록 좋음)
    avg_vals   = np.array([p.get("pitcher_vs_batter_avg", 0.250) for p in pool])
    pitch_vals = np.array([p.get("total_pitch_count", 0)         for p in pool])

    def normalize(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

    avg_score   = normalize(avg_vals)    # 낮을수록 좋음
    pitch_score = normalize(pitch_vals)  # 낮을수록 좋음
    combo_score = 0.5 * avg_score + 0.5 * pitch_score  # 균형형 종합 점수

    # 1번: 타자 상성 최우선 (avg 가장 낮은)
    idx1 = int(np.argmin(avg_vals))

    # 2번: 가장 신선한 투수 (pitch_count 가장 낮은)
    # idx1과 중복 방지
    pitch_score_copy = pitch_vals.copy().astype(float)
    pitch_score_copy[idx1] = np.inf
    idx2 = int(np.argmin(pitch_score_copy))

    # 3번: 균형형 — 위 두 명 제외 후 combo_score 중간값
    used = {idx1, idx2}
    remaining_idx = [i for i in range(len(pool)) if i not in used]
    if remaining_idx:
        rem_scores = [(i, combo_score[i]) for i in remaining_idx]
        rem_scores.sort(key=lambda x: x[1])
        # 중간값 선택
        mid = len(rem_scores) // 2
        idx3 = rem_scores[mid][0]
    else:
        idx3 = idx1  # fallback (풀이 2명 이하면 중복 허용)

    selected = [pool[idx1], pool[idx2], pool[idx3]]

    # 중복 제거 (혹시 같은 투수가 여러 기준에서 뽑힐 경우)
    seen = set()
    unique = []
    for p in selected:
        if p["pitcher_id"] not in seen:
            seen.add(p["pitcher_id"])
            unique.append(p)

    return unique


def build_action_set(
    current_pitcher: dict,
    pitcher_pool: list[dict],
    batter_hit_type: str = "R",
) -> list[dict]:
    """
    현재 투수 1명 + 불펜 3명 = 총 4명 행동 집합 반환.
    index 0 = 현재 투수 유지
    index 1~3 = 불펜 후보
    """
    bullpen = select_bullpen(
        pitcher_pool,
        current_pitcher_id=current_pitcher["pitcher_id"],
        batter_hit_type=batter_hit_type,
    )
    return [current_pitcher] + bullpen


# ──────────────────────────────────────────────
# 2. 상태 인코더  GameState + λ → 10차원
# ──────────────────────────────────────────────
BASE_STATE_MAP = {
    "0": 0, "1": 1, "2": 2, "3": 3,
    "12": 4, "13": 5, "23": 6, "123": 7,
}
HALF_MAP  = {"top": 0, "bot": 1}
INPUT_DIM = 10

def encode_state(state: GameState, lambda_risk: float) -> torch.Tensor:
    vec = [
        state.inning / 12.0,
        float(HALF_MAP.get(state.half, 0)),
        state.out_count / 2.0,
        BASE_STATE_MAP.get(state.base_state, 0) / 7.0,
        float(np.clip(state.score_diff_attacker / 10.0, -1, 1)),
        float(np.clip(state.total_pitch_count / 150.0, 0, 1)),
        float(np.clip(state.inning_pitch_count / 40.0, 0, 1)),
        float(state.pitcher_vs_batter_avg),
        float(state.batter_recent_avg),
        float(lambda_risk),
    ]
    return torch.tensor(vec, dtype=torch.float32)


# ──────────────────────────────────────────────
# 3. Q-Network
# ──────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────
# 4. Experience Replay
# ──────────────────────────────────────────────
@dataclass
class Transition:
    state_vec:      torch.Tensor
    reward:         float
    next_state_vec: torch.Tensor
    done:           bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────
# 5. DQN 에이전트
# ──────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        simulator: MonteCarloSimulator,
        lambda_risk: float = 0.5,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 300,
        batch_size: int = 32,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 10,
    ):
        self.sim         = simulator
        self.lam         = lambda_risk
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.target_freq = target_update_freq

        self.q_net      = QNetwork(INPUT_DIM, hidden_dim).to(DEVICE)
        self.target_net = QNetwork(INPUT_DIM, hidden_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.buffer  = ReplayBuffer(buffer_capacity)
        self.steps   = 0
        self.episode = 0

    def _eps(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps / self.eps_decay)

    def _make_state(self, base: GameState, c: dict) -> GameState:
        return GameState(
            inning=base.inning, half=base.half,
            out_count=base.out_count, base_state=base.base_state,
            score_diff_attacker=base.score_diff_attacker,
            total_pitch_count=c.get("total_pitch_count", 0),
            inning_pitch_count=c.get("inning_pitch_count", 0),
            pitcher_id=str(c["pitcher_id"]),
            batter_id=str(base.batter_id),
            batter_hit_type=base.batter_hit_type,
            pitcher_vs_batter_avg=c.get("pitcher_vs_batter_avg", 0.250),
            batter_recent_avg=base.batter_recent_avg,
        )

    def _greedy(self, base: GameState, action_set: list[dict]) -> int:
        self.q_net.eval()
        with torch.no_grad():
            q_vals = [
                self.q_net(
                    encode_state(self._make_state(base, c), self.lam)
                    .unsqueeze(0).to(DEVICE)
                ).item()
                for c in action_set
            ]
        self.q_net.train()
        return int(np.argmax(q_vals))

    def select_action(self, base: GameState, action_set: list[dict]) -> int:
        self.steps += 1
        if random.random() < self._eps():
            return random.randrange(len(action_set))
        return self._greedy(base, action_set)

    def _reward(self, state: GameState) -> float:
        r = self.sim.simulate(state)
        return -r["e_wpa"] - self.lam * r["var_wpa"]

    def _optimize(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch  = self.buffer.sample(self.batch_size)
        s_vec  = torch.stack([t.state_vec      for t in batch]).to(DEVICE)
        ns_vec = torch.stack([t.next_state_vec for t in batch]).to(DEVICE)
        rews   = torch.tensor([t.reward for t in batch],
                               dtype=torch.float32).to(DEVICE)
        dones  = torch.tensor([t.done   for t in batch],
                               dtype=torch.float32).to(DEVICE)

        q_cur = self.q_net(s_vec)
        with torch.no_grad():
            q_next   = self.target_net(ns_vec)
            q_target = rews + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_cur, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def run_episode(
        self,
        situations: list[GameState],
        pitcher_pool: list[dict],
    ) -> dict:
        """
        1 에피소드 실행.
        각 타석마다 build_action_set()으로 현재 투수 1 + 불펜 3 = 4명 구성
        → DQN이 4명 중 1명 선택
        """
        self.episode += 1
        total_reward = 0.0
        total_loss   = 0.0
        loss_cnt     = 0

        for i, base in enumerate(situations):
            # 현재 투수 = pitcher_pool에서 pitcher_id 기준으로 찾거나
            # 없으면 pool 첫 번째를 현재 투수로 가정
            current = next(
                (p for p in pitcher_pool
                 if p["pitcher_id"] == base.pitcher_id),
                pitcher_pool[0]
            )

            # 행동 집합: 현재 투수 1 + 불펜 3 = 총 4명
            action_set = build_action_set(
                current_pitcher=current,
                pitcher_pool=pitcher_pool,
                batter_hit_type=base.batter_hit_type,
            )
            if not action_set:
                continue

            action_idx   = self.select_action(base, action_set)
            chosen       = action_set[action_idx]
            state        = self._make_state(base, chosen)

            reward       = self._reward(state)
            total_reward += reward
            done         = (i == len(situations) - 1)

            if not done:
                next_base = situations[i + 1]
                next_c    = build_action_set(
                    current_pitcher=chosen,
                    pitcher_pool=pitcher_pool,
                    batter_hit_type=next_base.batter_hit_type,
                )
                next_c    = next_c[0] if next_c else chosen
                next_s    = self._make_state(next_base, next_c)
                next_vec  = encode_state(next_s, self.lam)
            else:
                next_vec = torch.zeros(INPUT_DIM)

            self.buffer.push(Transition(
                encode_state(state, self.lam), reward, next_vec, done
            ))

            loss = self._optimize()
            if loss is not None:
                total_loss += loss
                loss_cnt   += 1

        if self.episode % self.target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "total_reward": total_reward,
            "avg_loss":     total_loss / loss_cnt if loss_cnt else 0.0,
            "epsilon":      self._eps(),
        }

    def save(self, path: str):
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps":      self.steps,
            "episode":    self.episode,
            "lambda":     self.lam,
        }, path)
        print(f"저장: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps   = ckpt["steps"]
        self.episode = ckpt["episode"]
        self.lam     = ckpt["lambda"]
        print(f"로드: {path}  (episode={self.episode})")


# ──────────────────────────────────────────────
# 6. 데이터 로드 유틸
# ──────────────────────────────────────────────
def load_situations(
    parquet: str = "data_analysis/results/hsk_pa_with_wpa.parquet",
    n: int = 500,
    seed: int = 42,
) -> list[GameState]:
    df = pd.read_parquet(parquet)
    df = df[df["data_quality_flag"] == ""].copy()
    df["pitcher_vs_batter_avg"] = df["pitcher_vs_batter_avg"].fillna(0.250)
    df["batter_recent_avg"]     = df["batter_recent_avg"].fillna(0.250)
    df = df.sample(n=min(n, len(df)), random_state=seed)

    return [
        GameState(
            inning=int(r["inning"]),
            half=str(r["half"]),
            out_count=int(r["out_count"]),
            base_state=str(r["base_state"]),
            score_diff_attacker=int(r["score_diff_attacker"]),
            total_pitch_count=int(r.get("total_pitch_count", 0)),
            inning_pitch_count=int(r.get("inning_pitch_count", 0)),
            pitcher_id=str(int(r["pitcher_id"])),
            batter_id=str(int(r["batter_id"])),
            batter_hit_type=str(r["batter_hit_type"]),
            pitcher_vs_batter_avg=float(r["pitcher_vs_batter_avg"]),
            batter_recent_avg=float(r["batter_recent_avg"]),
        )
        for _, r in df.iterrows()
    ]


def load_pitcher_pool(
    parquet: str = "data_analysis/results/hsk_pa_with_wpa.parquet",
    top_n: int = 15,   # 20 → 15 (select_bullpen에서 4명으로 압축)
) -> list[dict]:
    df = pd.read_parquet(parquet)
    df = df[df["data_quality_flag"] == ""].copy()
    df["pitcher_vs_batter_avg"] = df["pitcher_vs_batter_avg"].fillna(0.250)

    pool = (
        df.groupby("pitcher_id")
        .agg(
            avg=("pitcher_vs_batter_avg", "mean"),
            cnt=("pitcher_id", "count"),
            tp=("total_pitch_count", "mean"),
            ip=("inning_pitch_count", "mean"),
        )
        .sort_values("cnt", ascending=False)
        .head(top_n)
        .reset_index()
    )

    return [
        {
            "pitcher_id":            str(int(r["pitcher_id"])),
            "pitcher_vs_batter_avg": float(r["avg"]),
            "total_pitch_count":     int(r["tp"]),
            "inning_pitch_count":    int(r["ip"]),
        }
        for _, r in pool.iterrows()
    ]


# ──────────────────────────────────────────────
# 7. 학습 루프
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("KBO DQN 에이전트 학습")
    print("행동 공간: 현재 투수 1 + 불펜 3 = 총 4명")
    print("=" * 60)

    sim = MonteCarloSimulator(
        n_simulations=200,
        rng_seed=None,
    )

    print("\n데이터 로드...")
    all_situations = load_situations(n=500)
    pitcher_pool   = load_pitcher_pool(top_n=15)
    print(f"타석 상황: {len(all_situations)}개 | 투수 풀: {len(pitcher_pool)}명")

    # 불펜 선택 예시 출력
    print("\n[불펜 선택 예시]")
    ex_bullpen = select_bullpen(
        pitcher_pool,
        current_pitcher_id=pitcher_pool[0]["pitcher_id"],
    )
    print(f"  현재 투수: {pitcher_pool[0]['pitcher_id']}")
    labels = ["타자상성", "신선도", "균형형"]
    for label, p in zip(labels, ex_bullpen):
        print(f"  {label}: {p['pitcher_id']}  "
              f"(vs타율={p['pitcher_vs_batter_avg']:.3f}, "
              f"투구수={p['total_pitch_count']})")

    LAMBDA_LIST       = [0.0, 0.5, 1.0, 2.0]
    N_EPISODES        = 100
    SITUATIONS_PER_EP = 15

    for lam in LAMBDA_LIST:
        print(f"\n{'='*60}")
        label = {0.0:"공격적", 0.5:"균형", 1.0:"보수적", 2.0:"매우보수적"}[lam]
        print(f"λ={lam} ({label}) 에이전트 학습")
        print(f"{'='*60}")

        agent = DQNAgent(
            simulator=sim,
            lambda_risk=lam,
            hidden_dim=128,
            lr=1e-3,
            gamma=0.95,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=300,
            batch_size=32,
            buffer_capacity=5_000,
            target_update_freq=10,
        )

        history = []
        for ep in range(1, N_EPISODES + 1):
            sits   = random.sample(all_situations,
                                   min(SITUATIONS_PER_EP, len(all_situations)))
            result = agent.run_episode(sits, pitcher_pool)
            history.append(result)

            if ep % 5 == 0 or ep == 1:
                print(f"  ep {ep:3d}/{N_EPISODES} | "
                      f"reward={result['total_reward']:+.4f} | "
                      f"loss={result['avg_loss']:.5f} | "
                      f"ε={result['epsilon']:.3f}")

        agent.save(f"dqn_output/dqn_agent_lambda{lam}.pth")

        with open(f"dqn_output/history_lambda{lam}.json", "w") as f:
            json.dump(history, f, indent=2)

    meta = {
        "lambda_list":        LAMBDA_LIST,
        "pitcher_pool":       pitcher_pool,
        "input_dim":          INPUT_DIM,
        "hidden_dim":         128,
        "n_episodes":         N_EPISODES,
        "action_set_size":    4,
        "bullpen_selection":  {
            "1번": "pitcher_vs_batter_avg 최소 (타자 상성)",
            "2번": "total_pitch_count 최소 (신선도)",
            "3번": "두 조건 종합 중간값 (균형형)",
            "피로도 하드컷": "total_pitch_count >= 120 제외",
        },
    }
    with open("dqn_output/dqn_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ 학습 완료! dqn_output/ 확인")
    print("   다음: python dqn_recommend.py")
    print("=" * 60)