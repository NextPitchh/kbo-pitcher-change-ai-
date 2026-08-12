"""
KBO WPA Pipeline — DQN 실전 투수 추천

학습된 dqn_agent_lambda{λ}.pth 로드 →
현재 게임 상황에서 감독 성향(λ) 반영 최적 투수 추천
행동 공간: 현재 투수 1 + 불펜 3 = 총 4명

실행: python dqn_recommend.py
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from monte_carlo_simulate import MonteCarloSimulator, GameState
from dqn_agent import (
    DQNAgent, QNetwork, encode_state,
    INPUT_DIM, DEVICE,
    build_action_set, load_pitcher_pool,
)


def load_agent(lambda_risk: float) -> DQNAgent:
    sim = MonteCarloSimulator(n_simulations=1000, rng_seed=42)
    agent = DQNAgent(simulator=sim, lambda_risk=lambda_risk)
    agent.load(f"dqn_output/dqn_agent_lambda{lambda_risk}.pth")
    agent.q_net.eval()
    return agent


def recommend(
    game_state: GameState,
    current_pitcher: dict,
    pitcher_pool: list[dict],
    lambda_risk: float = 0.5,
) -> pd.DataFrame:
    """
    현재 게임 상황 → 4명 행동 집합 구성 → DQN Q값 + Monte Carlo 추천.

    index 0 = 현재 투수 유지
    index 1~3 = 불펜 후보 (타자상성 / 신선도 / 균형형)
    """
    agent = load_agent(lambda_risk)
    sim   = agent.sim

    action_set = build_action_set(
        current_pitcher=current_pitcher,
        pitcher_pool=pitcher_pool,
        batter_hit_type=game_state.batter_hit_type,
    )

    rows = []
    for i, c in enumerate(action_set):
        state = agent._make_state(game_state, c)

        # DQN Q값
        vec = encode_state(state, lambda_risk).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_val = agent.q_net(vec).item()

        # Monte Carlo 정밀 추정
        mc = sim.simulate(state)
        u  = -mc["e_wpa"] - lambda_risk * mc["var_wpa"]

        role = {0: "현재투수(유지)", 1: "타자상성", 2: "신선도", 3: "균형형"}.get(i, "-")

        rows.append({
            "역할":              role,
            "pitcher_id":       c["pitcher_id"],
            "Q값(DQN)":         round(q_val, 5),
            "E[WPA]수비":       round(-mc["e_wpa"], 5),
            "Var[WPA]":         round(mc["var_wpa"], 6),
            f"U(λ={lambda_risk})": round(u, 5),
            "95%CI":            f"[{mc['ci_95'][0]:+.3f}, {mc['ci_95'][1]:+.3f}]",
            "vs타율":            round(c.get("pitcher_vs_batter_avg", 0.250), 3),
            "누적투구":          c.get("total_pitch_count", 0),
        })

    df = (pd.DataFrame(rows)
            .sort_values("Q값(DQN)", ascending=False)
            .reset_index(drop=True))
    df.insert(0, "순위", df.index + 1)

    lam_label = {0.0:"공격적", 0.5:"균형",
                 1.0:"보수적", 2.0:"매우보수적"}.get(lambda_risk, f"λ={lambda_risk}")
    situation = (f"{game_state.inning}회 {game_state.half}, "
                 f"{game_state.out_count}아웃, {game_state.base_state}루, "
                 f"점수차={game_state.score_diff_attacker:+d}")

    print(f"\n{'='*60}")
    print(f"투수 추천 | {situation}")
    print(f"감독 성향: {lam_label} (λ={lambda_risk})")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    best = df.iloc[0]
    action = "유지" if best["역할"] == "현재투수(유지)" else f"교체 → {best['pitcher_id']}"
    print(f"\n★ 추천: {action}  "
          f"(Q={best['Q값(DQN)']:.4f}, "
          f"U={best[f'U(λ={lambda_risk})']:.4f})")

    return df


def compare_by_lambda(
    game_state: GameState,
    current_pitcher: dict,
    pitcher_pool: list[dict],
    lambda_list: list[float] = [0.0, 0.5, 1.0, 2.0],
):
    """λ별 추천 결과 비교 — 감독 성향에 따른 전략 차이."""
    label = {
        0.0: "공격적 (λ=0.0)",
        0.5: "균형   (λ=0.5)",
        1.0: "보수적 (λ=1.0)",
        2.0: "매우보수(λ=2.0)",
    }

    print(f"\n{'='*60}")
    print("감독 성향(λ)별 추천 비교")
    print(f"{'='*60}")

    for lam in lambda_list:
        try:
            df   = recommend(game_state, current_pitcher,
                             pitcher_pool, lambda_risk=lam)
            best = df.iloc[0]
            action = ("유지" if best["역할"] == "현재투수(유지)"
                      else f"교체({best['역할']})")
            print(f"  {label.get(lam, f'λ={lam}'):17s} → "
                  f"{action:12s} | {best['pitcher_id']:8s} | "
                  f"Q={best['Q값(DQN)']:+.4f}")
        except FileNotFoundError:
            print(f"  λ={lam}: 모델 없음 — dqn_agent.py 먼저 실행")


# ──────────────────────────────────────────────
# 실행 데모
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("KBO DQN 투수 추천 시스템")
    print("=" * 60)

    # 실제 데이터에서 투수 풀 로드
    pitcher_pool = load_pitcher_pool(top_n=15)

    # 현재 게임 상황
    current_state = GameState(
        inning=7, half="bot",
        out_count=1, base_state="1",
        score_diff_attacker=-1,
        pitcher_id=pitcher_pool[0]["pitcher_id"],
        batter_id="88888",
        batter_hit_type="R",
        batter_recent_avg=0.310,
        pitcher_vs_batter_avg=pitcher_pool[0]["pitcher_vs_batter_avg"],
        total_pitch_count=pitcher_pool[0]["total_pitch_count"],
    )

    current_pitcher = pitcher_pool[0]

    print(f"\n현재 투수: {current_pitcher['pitcher_id']}  "
          f"(vs타율={current_pitcher['pitcher_vs_batter_avg']:.3f}, "
          f"누적투구={current_pitcher['total_pitch_count']}구)")

    # λ=0.5 기준 상세 추천
    recommend(current_state, current_pitcher, pitcher_pool, lambda_risk=0.5)

    # λ별 비교
    compare_by_lambda(current_state, current_pitcher, pitcher_pool)

    print("\n✅ 추천 완료!")