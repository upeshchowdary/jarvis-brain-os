"""Gateway operation for the autonomous browser agent."""

from __future__ import annotations

from automation.gateway import OpResult, Preview, Risk, gateway


def _preview(args) -> Preview:
    where = f" starting at {args['start_url']}" if args.get("start_url") else ""
    return Preview(f"Operate the browser{where} to: {args.get('goal')} (I'll stop before any purchase, send or delete, and never type passwords)")


@gateway.operation("browser.agent", Risk.MEDIUM, "Let JARVIS operate the browser for a goal", preview=_preview)
async def browser_agent_op(goal: str, start_url: str = "") -> OpResult:
    from automation.browser_agent import browser_agent

    try:
        outcome = await browser_agent.run(goal, start_url)
    except Exception as exc:
        return OpResult.fail(f"The browser agent couldn't start: {exc}")
    return OpResult(outcome.success, outcome.message, data={"steps": [s.__dict__ for s in outcome.steps], "needs_user": outcome.needs_user})
