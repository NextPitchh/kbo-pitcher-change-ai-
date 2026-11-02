"""PA API contract: native model labels, without an unsupported OUT/SF split."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, FiniteFloat, StrictInt, StrictStr, Field


class PaFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    inning: StrictInt = Field(ge=1, le=12)
    score_diff_attacker: StrictInt
    out_count: StrictInt = Field(ge=0, le=2)
    total_pitch_count: StrictInt = Field(ge=0)
    inning_pitch_count: StrictInt = Field(ge=0)
    pitcher_vs_batter_avg: FiniteFloat | None = Field(ge=0, le=1)
    batter_recent_avg: FiniteFloat | None = Field(ge=0, le=1)
    half: Literal["top", "bot"]
    base_state: Literal["0", "1", "2", "3", "12", "13", "23", "123"]
    batter_hit_type: StrictStr | None
    pitcher_id: StrictStr | StrictInt | None
    batter_id: StrictStr | StrictInt | None


class PaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: PaFeatures
