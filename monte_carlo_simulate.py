"""
KBO WPA Pipeline — Monte Carlo 시뮬레이션 (v5 클래스 기준)

클래스 9개: 1B / 2B / 3B / HR / BB / SO / OUT / GDP / SF
GDP: 아웃 +2, 1루 주자 제거 (병살타)
SF : 아웃 +1, 3루 주자 홈 득점 (희생플라이)

실행: python monte_carlo_simulate.py
요구사항: pip install catboost pandas numpy pyarrow
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from catboost import CatBoostClassifier

# ──────────────────────────────────────────────
# 0. WE 조회 모듈
# ──────────────────────────────────────────────
try:
    from data_analysis.methods.we_re_lookup import get_we_with_boundary
    print("✅ we_re_lookup 로드 성공")
except ImportError:
    print("⚠️  we_re_lookup 임포트 실패 — fallback WE 사용 (테스트용)")

    def get_we_with_boundary(inning: int, half: str, score_diff: int,
                              out_count: int, base_state: str) -> float:
        if inning >= 9 and half == "bot" and score_diff > 0:
            return 1.0
        if inning >= 9 and half == "bot" and score_diff == 0:
            return 0.65
        base = 0.5 + np.clip(score_diff * 0.07, -0.45, 0.45)
        return float(np.clip(base, 0.02, 0.98))


# ──────────────────────────────────────────────
# 1. 게임 상태
# ──────────────────────────────────────────────
@dataclass
class GameState:
    """
    한 타석 시작 시점 게임 상태.
    HANDOFF.md 컨벤션:
      score_diff_attacker: 공격팀 관점 (양수=리드, 추가 변환 불필요)
      half: "top"=초(원정공격) / "bot"=말(홈공격)
    """
    inning: int
    half: str
    out_count: int
    base_state: str           # "0","1","2","3","12","13","23","123"
    score_diff_attacker: int
    total_pitch_count: int = 0
    inning_pitch_count: int = 0
    pitcher_id: str = "unknown"
    batter_id: str = "unknown"
    batter_hit_type: str = "R"
    pitcher_vs_batter_avg: float = 0.250
    batter_recent_avg: float = 0.250

    def to_feature_row(self) -> dict:
        return {
            "inning":                self.inning,
            "score_diff_attacker":   self.score_diff_attacker,
            "out_count":             self.out_count,
            "total_pitch_count":     self.total_pitch_count,
            "inning_pitch_count":    self.inning_pitch_count,
            "pitcher_vs_batter_avg": self.pitcher_vs_batter_avg,
            "batter_recent_avg":     self.batter_recent_avg,
            "half":                  str(self.half),
            "base_state":            str(self.base_state),
            "batter_hit_type":       str(self.batter_hit_type),
            "pitcher_id":            str(self.pitcher_id),
            "batter_id":             str(self.batter_id),
        }


# ──────────────────────────────────────────────
# 2. 루상 비트 연산 유틸
#    1루=bit0, 2루=bit1, 3루=bit2
# ──────────────────────────────────────────────
BASE_STR_TO_BITS = {
    "0": 0b000, "1": 0b001, "2": 0b010, "3": 0b100,
    "12": 0b011, "13": 0b101, "23": 0b110, "123": 0b111,
}
BITS_TO_BASE_STR = {v: k for k, v in BASE_STR_TO_BITS.items()}

def _b(s: str) -> int:
    return BASE_STR_TO_BITS.get(s, 0)

def _s(b: int) -> str:
    return BITS_TO_BASE_STR.get(b & 0b111, "0")


# ──────────────────────────────────────────────
# 3. 상태 전이 엔진
# ──────────────────────────────────────────────
def apply_pa_result(
    state: GameState, pa_result: str
) -> tuple[GameState, int, bool]:
    """
    PA 결과 적용 → (다음 상태, 득점, 이닝종료여부)

    클래스별 처리 (v5 기준):
      1B  : 타자 1루, 주자 1루씩 진루 (3루→홈)
      2B  : 타자 2루, 1루→3루, 2루·3루→홈
      3B  : 타자 3루, 전 주자 홈
      HR  : 타자+전주자 홈
      BB  : 타자 1루, 꽉 찬 경우만 밀어내기 득점
      SO  : 아웃 +1, 주자 변화 없음
      OUT : 아웃 +1, 주자 변화 없음
      GDP : 아웃 +2, 1루 주자 제거 (병살타)
      SF  : 아웃 +1, 3루 주자 홈 득점 (희생플라이)
    """
    bits = _b(state.base_state)
    runs = 0
    new_out  = state.out_count
    new_bits = bits

    if pa_result == "HR":
        runs = bin(bits).count("1") + 1  # 전주자 + 타자
        new_bits = 0b000

    elif pa_result == "3B":
        runs = bin(bits).count("1")      # 전주자 홈
        new_bits = 0b100                 # 타자 3루

    elif pa_result == "2B":
        # 3루→홈, 2루→홈, 1루→3루
        if bits & 0b100: runs += 1
        if bits & 0b010: runs += 1
        new_bits = 0b010                 # 타자 2루
        if bits & 0b001: new_bits |= 0b100  # 1루→3루

    elif pa_result == "1B":
        # 3루→홈, 2루→3루, 1루→2루, 타자→1루
        if bits & 0b100: runs += 1
        new_bits = 0b001                 # 타자 1루
        if bits & 0b010: new_bits |= 0b100  # 2루→3루
        if bits & 0b001: new_bits |= 0b010  # 1루→2루

    elif pa_result == "BB":
        if bits == 0b111:                # 만루 → 밀어내기
            runs = 1
            new_bits = 0b111
        else:
            # 타자 1루, 연결된 주자만 밀려남
            if (bits & 0b011) == 0b011:  # 1·2루 → 타자+1루→2루, 2루→3루
                new_bits = (bits & 0b100) | 0b011
                new_bits = (bits & ~0b011) | 0b111  # 1·2루→2·3루 + 타자1루
                new_bits = 0b111 & ~(0 if bits & 0b100 else 0b100)
                # 명확하게 재계산
                new_bits = bits | 0b001  # 타자 1루 추가
                if bits & 0b001:         # 기존 1루 주자 2루로
                    new_bits = (new_bits & ~0b001) | 0b010 | 0b001
                    if bits & 0b010:     # 기존 2루 주자 3루로
                        new_bits |= 0b100
            elif bits & 0b001:           # 1루만 → 1루→2루, 타자→1루
                new_bits = (bits & ~0b001) | 0b010 | 0b001
            else:
                new_bits = bits | 0b001  # 타자 1루만 추가

    elif pa_result in ("SO", "OUT"):
        new_out += 1
        new_bits = bits                  # 주자 변화 없음

    elif pa_result == "GDP":
        # 병살: 아웃 +2, 1루 주자 제거
        # 1루 주자 없으면 그냥 아웃 +2로 처리 (희귀 케이스)
        new_out  += 2
        new_bits  = bits & ~0b001        # 1루 주자 제거

    elif pa_result == "SF":
        # 희생플라이: 아웃 +1, 3루 주자 홈
        new_out += 1
        if bits & 0b100:                 # 3루 주자 있으면 득점
            runs    += 1
            new_bits = bits & ~0b100     # 3루 주자 제거
        # 3루 주자 없어도 SF는 아웃 +1 (데이터 노이즈 케이스)

    inning_ended = new_out >= 3

    next_state = GameState(
        inning=state.inning,
        half=state.half,
        out_count=0 if inning_ended else new_out,
        base_state="0" if inning_ended else _s(new_bits),
        score_diff_attacker=state.score_diff_attacker,
        total_pitch_count=state.total_pitch_count + 1,
        inning_pitch_count=0 if inning_ended else state.inning_pitch_count + 1,
        pitcher_id=state.pitcher_id,
        batter_id=state.batter_id,
        batter_hit_type=state.batter_hit_type,
        pitcher_vs_batter_avg=state.pitcher_vs_batter_avg,
        batter_recent_avg=state.batter_recent_avg,
    )
    return next_state, runs, inning_ended


def next_half_inning(state: GameState, abs_score_diff: int) -> GameState:
    """
    반이닝 전환.
    abs_score_diff: home - away 절대 점수차
    """
    if state.half == "top":
        new_half   = "bot"
        new_inning = state.inning
        new_sda    = abs_score_diff          # 홈팀 공격: score_diff_attacker = home-away
    else:
        new_half   = "top"
        new_inning = state.inning + 1
        new_sda    = -abs_score_diff         # 원정팀 공격: 부호 반전

    return GameState(
        inning=new_inning,
        half=new_half,
        out_count=0,
        base_state="0",
        score_diff_attacker=new_sda,
        total_pitch_count=0,
        inning_pitch_count=0,
        pitcher_id=state.pitcher_id,
        batter_id=state.batter_id,
        batter_hit_type=state.batter_hit_type,
        pitcher_vs_batter_avg=state.pitcher_vs_batter_avg,
        batter_recent_avg=state.batter_recent_avg,
    )


# ──────────────────────────────────────────────
# 4. 몬테카를로 시뮬레이터
# ──────────────────────────────────────────────
class MonteCarloSimulator:

    def __init__(
        self,
        model_path: str = "catboost_output/catboost_pa_classifier.cbm",
        meta_path:  str = "catboost_output/model_meta.json",
        n_simulations: int = 1000,
        max_innings: int = 12,
        rng_seed: Optional[int] = 42,
    ):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        self.classes      = self.meta["classes"]
        self.cat_features = self.meta["cat_features"]
        self.all_features = self.meta["all_features"]
        self.n_simulations = n_simulations
        self.max_innings   = max_innings
        self.rng = np.random.default_rng(rng_seed)

        print(f"모델 로드: {model_path}")
        print(f"클래스 {len(self.classes)}개: {self.classes}")
        print(f"시뮬레이션: {n_simulations:,}회 (Batch Mode 최적화)")

    def _sample(self, proba: np.ndarray) -> str:
        idx = self.rng.choice(len(self.classes), p=proba)
        return self.classes[idx]

    def simulate(self, initial_state: GameState) -> dict:
        """Batch(배치) 예측을 적용하여 n회 시뮬레이션을 동시에 진행"""
        n = self.n_simulations
        
        # 1. 200개의 평행우주(초기 상태) 리스트 생성
        # active_indices: 아직 경기가 끝나지 않은 평행우주의 인덱스 추적
        active_indices = list(range(n))
        states = [GameState(**initial_state.__dict__) for _ in range(n)]
        
        we_start = get_we_with_boundary(
            inning=initial_state.inning, half=initial_state.half,
            score_diff=initial_state.score_diff_attacker,
            out_count=initial_state.out_count, base_state=initial_state.base_state,
        )
        
        # 홈팀 - 원정팀 점수차 절대값 추적
        init_abs_diff = initial_state.score_diff_attacker if initial_state.half == "bot" else -initial_state.score_diff_attacker
        abs_diffs = [init_abs_diff] * n
        
        # 각 평행우주별 최종 WE 결과 저장용
        we_ends = [0.0] * n
        
        loop_count = 0
        # 2. 살아있는 시뮬레이션이 있는 동안 타석을 동시 진행
        while active_indices and loop_count < 300:
            loop_count += 1
            
            # --- [핵심] 살아있는 상태들만 모아서 Batch(2차원 배열) 구성 ---
            batch_features = []
            for idx in active_indices:
                feat = states[idx].to_feature_row()
                row = [str(feat.get(c, 0)) if c in self.cat_features else float(feat.get(c, 0)) for c in self.all_features]
                batch_features.append(row)
                
            # CatBoost 모델로 200명의 타석 결과를 단 1번의 연산으로 예측!
            probas_batch = self.model.predict_proba(batch_features)
            
            next_active = []
            
            # --- 예측된 결과를 바탕으로 각각의 평행우주 업데이트 ---
            for i, proba in enumerate(probas_batch):
                idx = active_indices[i]
                state = states[idx]
                
                proba = proba / proba.sum() # 수치 안정성
                pa_result = self._sample(proba)
                
                next_state, runs, inning_ended = apply_pa_result(state, pa_result)
                
                if state.half == "bot":
                    abs_diffs[idx] += runs
                else:
                    abs_diffs[idx] -= runs
                    
                is_done = False
                
                if inning_ended:
                    if state.inning >= 9 and state.half == "bot" and abs_diffs[idx] > 0:
                        we_ends[idx] = 1.0
                        is_done = True
                    else:
                        next_state = next_half_inning(state, abs_diffs[idx])
                        if next_state.inning > self.max_innings:
                            is_done = True
                            
                if is_done:
                    # 경기가 종료된 평행우주는 최종 WE 계산 후 active 목록에서 제외됨
                    if we_ends[idx] == 0.0: 
                        we_ends[idx] = get_we_with_boundary(
                            inning=min(next_state.inning, self.max_innings),
                            half=next_state.half,
                            score_diff=np.clip(next_state.score_diff_attacker, -10, 10),
                            out_count=min(next_state.out_count, 2),
                            base_state=next_state.base_state,
                        )
                else:
                    states[idx] = next_state
                    next_active.append(idx) # 경기가 안 끝났으면 다음 타석 진행을 위해 추가
                    
            active_indices = next_active
            
        # (무한루프 방지) 300타석 넘게 끝나지 않은 잔여 게임 강제 WE 계산
        for idx in active_indices:
            s = states[idx]
            we_ends[idx] = get_we_with_boundary(
                inning=min(s.inning, self.max_innings),
                half=s.half,
                score_diff=np.clip(s.score_diff_attacker, -10, 10),
                out_count=min(s.out_count, 2),
                base_state=s.base_state,
            )
            
        wpa_samples = np.array(we_ends) - we_start
        
        return {
            "e_wpa":   float(np.mean(wpa_samples)),
            "var_wpa": float(np.var(wpa_samples)),
            "std_wpa": float(np.std(wpa_samples)),
            "ci_95":   (float(np.percentile(wpa_samples, 2.5)),
                        float(np.percentile(wpa_samples, 97.5))),
            "wpa_dist": wpa_samples.tolist(),
            "n_sim":   self.n_simulations,
        }

    def compare_pitchers(
        self,
        base_state: GameState,
        pitcher_candidates: list[dict],
        lambda_risk: float = 0.5,
    ) -> pd.DataFrame:
        """
        투수 후보별 시뮬레이션 → 마코위츠 효용 U 비교.
        """
        rows = []
        for c in pitcher_candidates:
            state = GameState(
                inning=base_state.inning, half=base_state.half,
                out_count=base_state.out_count, base_state=base_state.base_state,
                score_diff_attacker=base_state.score_diff_attacker,
                total_pitch_count=c.get("total_pitch_count", 0),
                inning_pitch_count=c.get("inning_pitch_count", 0),
                pitcher_id=str(c["pitcher_id"]), batter_id=str(base_state.batter_id),
                batter_hit_type=base_state.batter_hit_type,
                pitcher_vs_batter_avg=c.get("pitcher_vs_batter_avg", 0.250),
                batter_recent_avg=base_state.batter_recent_avg,
            )
            r = self.simulate(state)

            e_wpa_p  = -r["e_wpa"]          # 투수 관점 부호 반전
            var_wpa  =  r["var_wpa"]         # 분산은 부호 무관
            utility  = e_wpa_p - lambda_risk * var_wpa

            rows.append({
                "pitcher_id": c["pitcher_id"],
                "E[WPA]":     round(e_wpa_p, 5),
                "Var[WPA]":   round(var_wpa, 6),
                "Std[WPA]":   round(r["std_wpa"], 5),
                f"U(λ={lambda_risk})": round(utility, 5),
                "CI_95":      f"[{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]",
            })

        return (pd.DataFrame(rows)
                  .sort_values(f"U(λ={lambda_risk})", ascending=False)
                  .reset_index(drop=True)
                  .assign(rank=lambda d: d.index + 1))


# ──────────────────────────────────────────────
# 5. 실행 데모
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("KBO Monte Carlo 시뮬레이터 — 데모")
    print("=" * 60)

    sim = MonteCarloSimulator(n_simulations=1000, rng_seed=42)

    # ── 데모 1: 단일 상태 ──
    print("\n[데모 1] 단일 게임 상태 WPA 분포")
    print("-" * 40)
    state = GameState(
        inning=7, half="bot",
        out_count=1, base_state="1",
        score_diff_attacker=-1,
        pitcher_id="99999", batter_id="88888",
        batter_hit_type="R",
        pitcher_vs_batter_avg=0.280,
        batter_recent_avg=0.310,
    )
    print(f"상태: {state.inning}회 {state.half}, {state.out_count}아웃, "
          f"{state.base_state}루, 점수차={state.score_diff_attacker:+d}")

    r = sim.simulate(state)
    print(f"\n결과 (n={r['n_sim']:,}회):")
    print(f"  E[WPA]   = {r['e_wpa']:+.5f}")
    print(f"  Var[WPA] = {r['var_wpa']:.6f}")
    print(f"  Std[WPA] = {r['std_wpa']:.5f}")
    print(f"  95% CI   = [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")

    print(f"\n감독 성향(λ)별 DQN 효용 U = E[WPA] - λ·Var[WPA]:")
    for lam in [0.0, 0.5, 1.0, 2.0]:
        u = r["e_wpa"] - lam * r["var_wpa"]
        tag = {0.0: "공격적", 0.5: "균형", 1.0: "보수적", 2.0: "매우보수적"}[lam]
        print(f"  λ={lam} ({tag:6s}): U = {u:+.5f}")

    # ── 데모 2: 투수 비교 ──
    print("\n[데모 2] 투수 후보 비교")
    print("-" * 40)
    candidates = [
        {"pitcher_id": "P001", "pitcher_vs_batter_avg": 0.220,
         "total_pitch_count": 40,  "inning_pitch_count": 8},
        {"pitcher_id": "P002", "pitcher_vs_batter_avg": 0.310,
         "total_pitch_count": 80,  "inning_pitch_count": 15},
        {"pitcher_id": "P003", "pitcher_vs_batter_avg": 0.265,
         "total_pitch_count": 20,  "inning_pitch_count": 3},
    ]

    for lam in [0.5, 1.0]:
        print(f"\n  λ={lam} 기준:")
        df = sim.compare_pitchers(state, candidates, lambda_risk=lam)
        print(df[["rank", "pitcher_id", "E[WPA]", "Var[WPA]",
                   f"U(λ={lam})", "CI_95"]].to_string(index=False))
        print(f"  → 추천: {df.iloc[0]['pitcher_id']}")

    print("\n✅ 완료! → 다음: python dqn_agent.py")