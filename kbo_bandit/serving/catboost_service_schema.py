from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator


class ServicePaState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    game_id: StrictStr = Field(pattern=r"^2025[0-9]{4}[A-Z]{4}[012]2025$")
    inning: StrictInt = Field(ge=1, le=9)
    half: Literal["top", "bot"]
    out_count: StrictInt = Field(ge=0, le=2)
    base_state: Literal["0", "1", "2", "3", "12", "13", "23", "123"]
    home_score: StrictInt = Field(ge=0)
    away_score: StrictInt = Field(ge=0)
    pitcher_id: StrictInt | StrictStr
    batter_id: StrictInt | StrictStr
    total_pitch_count_before: StrictInt = Field(ge=0)
    inning_pitch_count_before: StrictInt = Field(ge=0)

    @field_validator("pitcher_id", "batter_id")
    @classmethod
    def player_id(cls, value):
        s = str(value)
        if not s.isascii() or not s.isdigit() or s.startswith("0") or len(s) > 9:
            raise ValueError("player ID must be a positive canonical decimal identifier")
        return s

    @model_validator(mode="after")
    def pitch_counts(self):
        if self.inning_pitch_count_before > self.total_pitch_count_before:
            raise ValueError("inning pitch count exceeds total pitch count")
        return self


class ServicePaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: ServicePaState
