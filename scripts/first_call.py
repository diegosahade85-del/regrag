"""
Day 1 exercise: force Claude to return validated structured data instead of
free-form text, using tool_choice + a Pydantic schema.

Run:
    uv run python scripts/first_call.py
"""
import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-5"  # verify current id at docs.claude.com if this errors


class RegulatoryScope(BaseModel):
    country: str = Field(description="ISO country name")
    regulation_id: str = Field(description="Official identifier, e.g. 'Res. SIC 16/2025'")
    product_categories: list[str]
    obligations: list[str] = Field(description="Concrete obligations imposed")
    confidence: float = Field(ge=0, le=1)


TOOL = {
    "name": "record_scope",
    "description": "Record the regulatory scope extracted from the text.",
    "input_schema": RegulatoryScope.model_json_schema(),
}


def extract(text: str) -> RegulatoryScope:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_scope"},
        system=(
            "You extract regulatory scope from Latin American product "
            "compliance texts. Use only what is stated in the text. "
            "If a field is not stated, leave it empty and lower confidence."
        ),
        messages=[{"role": "user", "content": text}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return RegulatoryScope(**block.input)


if __name__ == "__main__":
    sample = open("data/raw/sample.txt", encoding="utf-8").read()[:8000]
    result = extract(sample)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
