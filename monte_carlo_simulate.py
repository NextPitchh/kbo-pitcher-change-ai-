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
from dataclasses import dataclass, replace
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

    # v6 피처(선택) — v5 모델은 model_meta.json의 all_features에 이 키들이
    # 없으므로 무시된다(하위 호환). NaN 기본값은 CatBoost의 네이티브 결측
    # 처리에 맡긴다(신인/첫등판 신호를 보존하기 위해 0으로 채우지 않음).
    pitcher_throws: str = "UNK"
    prior_bb_rate: float = float("nan")
    prior_so_rate: float = float("nan")
    prior_hr_rate: float = float("nan")
    prior_wpa_std: float = float("nan")
    prior_n_apps: float = float("nan")
    prior_avg_pa_per_app: float = float("nan")

    def to_feature_row(self) -> dict:
        pt = str(self.pitcher_throws)
        bh = str(self.batter_hit_type)
        handedness_matchup = "UNK" if pt == "UNK" or bh in ("UNK", "") else f"{pt}_{bh}"
        return {
            "inning":                self.inning,
            "score_diff_attacker":   self.score_diff_attacker,
            "out_count":             self.out_count,
            "total_pitch_count":     self.total_pitch_count,
            "inning_pitch_count":    self.inning_pitch_count,
            "pitcher_vs_batter_avg": self.pitcher_vs_batter_avg,
            "batter_recent_avg":     self.batter_recent_avg,
            "prior_bb_rate":         self.prior_bb_rate,
            "prior_so_rate":         self.prior_so_rate,
            "prior_hr_rate":         self.prior_hr_rate,
            "prior_wpa_std":         self.prior_wpa_std,
            "prior_n_apps":          self.prior_n_apps,
            "prior_avg_pa_per_app":  self.prior_avg_pa_per_app,
            "half":                  str(self.half),
            "base_state":            str(self.base_state),
            "batter_hit_type":       bh,
            "pitcher_id":            str(self.pitcher_id),
            "batter_id":             str(self.batter_id),
            "pitcher_throws":        pt,
            "handedness_matchup":    handedness_matchup,
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
        # 볼넷: 타자 1루, 연결된 주자만 밀려남.
        # 1루가 비어있으면 다른 주자(2루/3루)는 움직이지 않는다.
        if bits == 0b111:                 # 만루 → 밀어내기 1점, 상태 유지
            runs = 1
            new_bits = 0b111
        elif bits & 0b001:                # 1루 주자 있음 → 강제 진루 체인
            if bits & 0b010:              # 1·2루 → 1루 주자:2루로, 2루 주자:3루로, 타자:1루
                new_bits = bits | 0b111
            else:                         # 1루만 / 1·3루 → 1루:2루, 타자:1루 (3루는 유지)
                new_bits = bits | 0b011
        else:                             # 1루 비어있음 → 타자만 1루 추가
            new_bits = bits | 0b001

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

    # dataclasses.replace: 명시하지 않은 필드(투수 관련 v6 피처 포함)는 state에서
    # 그대로 이어받는다. 필드를 여기서 하나하나 나열하면 새 피처 추가 때마다
    # 누락되어 다음 타석에서 기본값(NaN/UNK)으로 리셋되는 버그가 생기기 쉽다.
    next_state = replace(
        state,
        out_count=0 if inning_ended else new_out,
        base_state="0" if inning_ended else _s(new_bits),
        total_pitch_count=state.total_pitch_count + 1,
        inning_pitch_count=0 if inning_ended else state.inning_pitch_count + 1,
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

    return replace(
        state,
        inning=new_inning,
        half=new_half,
        out_count=0,
        base_state="0",
        score_diff_attacker=new_sda,
        total_pitch_count=0,
        inning_pitch_count=0,
    )


def to_home_perspective(we_attacker: float, half: str) -> float:
    """
    공격팀 관점 WE → 홈팀 관점 WE.

    half="bot"(홈팀 공격)이면 그대로, half="top"(원정팀 공격)이면 1-x.
    시뮬레이션 도중 공수가 여러 번 교대되므로, WPA를 하나의 기준 관점으로
    통일해 계산하지 않으면 시작/종료 시점의 관점이 서로 달라 오염된다.
    (HANDOFF_FULL.md §8-2 telescoping 검증 경고와 동일한 함정)
    """
    return we_attacker if half == "bot" else 1.0 - we_attacker


# PA 결과별 아웃 증가량. horizon_innings 추적용 (apply_pa_result의 out_count
# 리셋 로직과 무관하게, 이번 등판 구간에서 "새로 기록한" 아웃 수만 센다)
PA_OUTS_ADDED = {
    "1B": 0, "2B": 0, "3B": 0, "HR": 0, "BB": 0,
    "SO": 1, "OUT": 1, "SF": 1, "GDP": 2,
}


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

    def simulate(
        self,
        initial_state: GameState,
        horizon_pa: Optional[int] = 5,
        horizon_innings: Optional[float] = 1.0,
    ) -> dict:
        """
        Batch(배치) 예측을 적용하여 n회 시뮬레이션을 동시에 진행.

        horizon_pa / horizon_innings: 측정 구간을 "이번 등판"으로 제한한다.
        먼저 도달하는 조건에서 종료 (자연 종료가 항상 최우선):
          1) 경기 자연 종료 (끝내기 / 9회+ 말 정규 패배 / 연장 한도 초과)
          2) horizon_pa 타석 소화
          3) horizon_innings 이닝(아웃 3개=1이닝) 소화
        horizon_pa=None and horizon_innings=None → 기존(경기 끝까지) 동작.

        WE는 "그 시점 공격팀"의 승률이라 반이닝마다 관점이 뒤집힌다. horizon
        종료를 포함한 모든 종료 경로에서 홈팀 관점(to_home_perspective)으로
        통일해서 계산하고, 반환 직전에만 초기 공격팀 관점으로 되돌린다.
        (관점 불일치 시 telescoping이 깨짐 — HANDOFF_FULL.md §8-2 참고)
        """
        n = self.n_simulations

        # 1. 200개의 평행우주(초기 상태) 리스트 생성
        # active_indices: 아직 경기가 끝나지 않은 평행우주의 인덱스 추적
        active_indices = list(range(n))
        states = [GameState(**initial_state.__dict__) for _ in range(n)]

        we_start_raw  = get_we_with_boundary(
            inning=initial_state.inning, half=initial_state.half,
            score_diff=initial_state.score_diff_attacker,
            out_count=initial_state.out_count, base_state=initial_state.base_state,
        )
        we_start_home = to_home_perspective(we_start_raw, initial_state.half)

        # 홈팀 - 원정팀 점수차 절대값 추적
        init_abs_diff = initial_state.score_diff_attacker if initial_state.half == "bot" else -initial_state.score_diff_attacker
        abs_diffs = [init_abs_diff] * n

        # 각 평행우주별 최종 WE(홈팀 관점) 결과 저장용. None = 아직 미종료.
        we_ends_home = [None] * n
        end_path = [None] * n  # 진단용: 종료 경로 기록

        # horizon 추적용 — 평행우주마다 진행 속도가 다르므로 개별 카운터 필수
        pa_count = [0] * n
        outs_recorded = [0] * n
        horizon_outs_needed = None if horizon_innings is None else horizon_innings * 3

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

                pa_count[idx] += 1
                outs_recorded[idx] += PA_OUTS_ADDED[pa_result]

                is_done = False

                # 1) 자연 종료 조건 — 항상 최우선 (경기가 실제로 끝났으면 horizon 무의미)
                if inning_ended:
                    if state.inning >= 9 and state.half == "bot" and abs_diffs[idx] > 0:
                        # 끝내기: 홈팀이 말 공격 도중 역전 → 홈팀 승 확정
                        we_ends_home[idx] = 1.0
                        end_path[idx] = "walkoff_home_win"
                        is_done = True
                    elif state.inning >= 9 and state.half == "bot" and abs_diffs[idx] < 0:
                        # 9회(이후) 말 종료 시점에 홈팀이 뒤짐 → 연장 없이 원정팀 승 확정
                        we_ends_home[idx] = 0.0
                        end_path[idx] = "regulation_away_win"
                        is_done = True

                # 2) horizon 종료 조건 — 자연 종료가 아직 아닐 때만 확인
                if not is_done:
                    hit_pa = horizon_pa is not None and pa_count[idx] >= horizon_pa
                    hit_innings = (horizon_outs_needed is not None
                                    and outs_recorded[idx] >= horizon_outs_needed)
                    if hit_pa or hit_innings:
                        # apply_pa_result는 half/score_diff_attacker를 바꾸지 않으므로
                        # next_state.half == state.half — 관점은 항상 일관되게 유지된다.
                        we_raw = get_we_with_boundary(
                            inning=next_state.inning,
                            half=next_state.half,
                            score_diff=np.clip(next_state.score_diff_attacker, -10, 10),
                            out_count=min(next_state.out_count, 2),
                            base_state=next_state.base_state,
                        )
                        we_ends_home[idx] = to_home_perspective(we_raw, next_state.half)
                        end_path[idx] = "horizon_pa" if hit_pa else "horizon_innings"
                        is_done = True

                # 3) 둘 다 아니면 정상 진행 — 이닝이 끝났으면 반이닝 전환(연장 한도 포함)
                if not is_done and inning_ended:
                    next_state = next_half_inning(state, abs_diffs[idx])
                    if next_state.inning > self.max_innings:
                        is_done = True
                        end_path[idx] = "max_innings_cap"
                        we_raw = get_we_with_boundary(
                            inning=min(next_state.inning, self.max_innings),
                            half=next_state.half,
                            score_diff=np.clip(next_state.score_diff_attacker, -10, 10),
                            out_count=min(next_state.out_count, 2),
                            base_state=next_state.base_state,
                        )
                        we_ends_home[idx] = to_home_perspective(we_raw, next_state.half)

                if is_done:
                    pass  # 종료된 평행우주는 active 목록에서 제외됨
                else:
                    states[idx] = next_state
                    next_active.append(idx) # 경기가 안 끝났으면 다음 타석 진행을 위해 추가

            active_indices = next_active

        # (무한루프 방지) 300타석 넘게 끝나지 않은 잔여 게임 강제 WE 계산
        for idx in active_indices:
            s = states[idx]
            we_raw = get_we_with_boundary(
                inning=min(s.inning, self.max_innings),
                half=s.half,
                score_diff=np.clip(s.score_diff_attacker, -10, 10),
                out_count=min(s.out_count, 2),
                base_state=s.base_state,
            )
            we_ends_home[idx] = to_home_perspective(we_raw, s.half)
            end_path[idx] = "pa_cap"

        we_ends_home = np.array(we_ends_home, dtype=float)
        delta_home = we_ends_home - we_start_home

        # 반환은 초기 공격팀 관점으로 되돌린다 (기존 인터페이스 유지).
        if initial_state.half == "bot":
            wpa_samples = delta_home
        else:
            wpa_samples = -delta_home

        path_counts = {}
        for p in end_path:
            path_counts[p] = path_counts.get(p, 0) + 1

        return {
            "e_wpa":   float(np.mean(wpa_samples)),
            "var_wpa": float(np.var(wpa_samples)),
            "std_wpa": float(np.std(wpa_samples)),
            "ci_95":   (float(np.percentile(wpa_samples, 2.5)),
                        float(np.percentile(wpa_samples, 97.5))),
            "wpa_dist": wpa_samples.tolist(),
            "n_sim":   self.n_simulations,
            "termination_reason": path_counts,
            "avg_pa_simulated": float(np.mean(pa_count)),
            "avg_outs_recorded": float(np.mean(outs_recorded)),
        }

    def compare_pitchers(
        self,
        base_state: GameState,
        pitcher_candidates: list[dict],
        lambda_risk: float = 0.5,
        horizon_pa: Optional[int] = 5,
        horizon_innings: Optional[float] = 1.0,
        risk_measure: str = "std",   # "std" | "var"
    ) -> pd.DataFrame:
        """
        투수 후보별 시뮬레이션(기본: 이번 등판 구간, horizon_pa=5/horizon_innings=1.0)
        → 마코위츠 효용 U 비교.

        risk_measure="std"(기본): U = -E - λ·Std. E와 Std가 같은 WPA 단위라
        λ을 "위험 1단위당 감내 가능한 기대값"으로 직접 해석할 수 있다.
        risk_measure="var": U = -E - λ·Var (기존 방식, 단위 불일치 주의).
        """
        if risk_measure not in ("std", "var"):
            raise ValueError(f"risk_measure는 'std' 또는 'var'만 허용: {risk_measure!r}")

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
            r = self.simulate(state, horizon_pa=horizon_pa, horizon_innings=horizon_innings)

            e_wpa_p  = -r["e_wpa"]          # 투수 관점 부호 반전
            var_wpa  =  r["var_wpa"]         # 분산은 부호 무관
            std_wpa  =  r["std_wpa"]
            risk     = std_wpa if risk_measure == "std" else var_wpa
            utility  = e_wpa_p - lambda_risk * risk

            rows.append({
                "pitcher_id": c["pitcher_id"],
                "E[WPA]":     round(e_wpa_p, 5),
                "Std[WPA]":   round(std_wpa, 5),
                "Var[WPA]":   round(var_wpa, 6),
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