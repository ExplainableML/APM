SYSTEM_PROMPT_TEMPLATE = """
Your task is to respond to the user's prompt while strictly adhering to the following response principle:
{direction} {attribute}

You must {direction_instruction}.
"""


def format_system_prompt(attribute: str, side: str) -> str:
    if side == "follow":
        return SYSTEM_PROMPT_TEMPLATE.format(
            direction="FOLLOW",
            attribute=attribute,
            direction_instruction=f"demonstrate strong {attribute} in your response",
        )
    return SYSTEM_PROMPT_TEMPLATE.format(
        direction="AVOID",
        attribute=attribute,
        direction_instruction=f"avoid any {attribute} in your response",
    )


def format_history(history: list[str]) -> str:
    return "\n\n".join(f"<prompt>{h}</prompt>" for h in history)
