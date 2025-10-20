from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .framework import AngelusFramework, AngelusConfig
from .common.types import Intent, SigilSpec
from .common.sigil import generate_sigil_svg


class IntentRequest(BaseModel):
    user_id: Optional[str] = None
    text: str
    tags: list[str] = []


class GuidanceResponse(BaseModel):
    summary: str
    steps: list[str]
    confidence: float
    archetype: Optional[str] = None


app = FastAPI(title="Angelus API")
framework = AngelusFramework(AngelusConfig())


@app.post("/guidance", response_model=GuidanceResponse)
async def guidance(req: IntentRequest):
    intent = Intent(user_id=req.user_id, text=req.text, tags=req.tags)
    g = framework.run(intent)
    return GuidanceResponse(summary=g.summary, steps=g.steps, confidence=g.confidence, archetype=g.archetype)


@app.get("/sigil")
async def sigil(seed: str, color: str = "#4B6FFF", background: str = "#0B1020", size: int = 256, stroke: int = 3):
    svg = generate_sigil_svg(SigilSpec(seed=seed, color=color, background=background, size=size, stroke=stroke))
    return {
        "svg": svg,
    }
