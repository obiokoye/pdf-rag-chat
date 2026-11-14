"""A scripted, OpenAI-shaped client used by demo mode and by the tests.

It reads the numbered <source> blocks in the prompt, finds a script whose
keywords match the question, and writes that answer with citations pointing
at whichever source number holds the right page. If retrieval didn't find
that page, the citation is left out, so demo answers are only as well cited
as the retrieval really was. Unscripted questions get the closest passage.
"""

from __future__ import annotations

import re
from types import SimpleNamespace as NS
from typing import Any

_SOURCE = re.compile(r'<source n="(\d+)" doc="([^"]+)" page="(\d+)">\n(.*?)\n</source>', re.S)
_REF = re.compile(r"\{(\w[\w-]*):(\d+)\}")

# (keywords: any one must appear in the question, answer with {doc-prefix:page} citations)
Script = tuple[tuple[str, ...], str]

DEMO_SCRIPTS: list[Script] = [
    (
        ("carry", "annual leave", "holiday"),
        "You get 20 days of paid annual leave per calendar year, plus public holidays {brightline:3}. "
        "You can carry up to 5 unused days into the next year, but they must be used by 31 March or "
        "they are lost {brightline:3}.",
    ),
    (
        ("sick",),
        "Sick leave is up to 10 paid days per year {brightline:3}. Tell your manager before 10:00 on the "
        "first day, and bring a doctor's note after 2 consecutive sick days {brightline:3}.",
    ),
    (
        ("uptime", "credit", "downtime", "sla"),
        "NimbusHost commits to 99.9% monthly uptime per server {nimbushost:2}. Below that you get a 10% "
        "service credit, rising to 25% below 99.0% and 50% below 95.0% {nimbushost:2}. You have to open a "
        "support ticket within 30 days of the end of the month, and credits go on the next invoice, never "
        "as cash {nimbushost:2}.",
    ),
    (
        ("backup", "restore"),
        "Every server gets a daily snapshot backup that is kept for 14 days, and restore requests are "
        "answered within 4 business hours {nimbushost:3}. The agreement still makes you responsible for "
        "keeping your own copies of important data {nimbushost:3}.",
    ),
    (
        ("expense", "meal", "receipt", "reimburse"),
        "Submit claims in the finance portal within 30 days, with a receipt for every item over NGN 5,000 "
        "{brightline:4}. Meals while travelling are covered up to NGN 15,000 per day, and anything above "
        "NGN 100,000 needs your manager's approval first {brightline:4}.",
    ),
    (
        ("remote", "office", "hybrid", "core hours"),
        "Brightline is hybrid: Tuesday to Thursday in the office, Monday and Friday remote {brightline:2}. "
        "Core hours are 10:00 to 16:00 Lagos time, and working fully remote for more than two weeks needs "
        "written approval {brightline:2}.",
    ),
    (
        ("cancel", "terminat", "refund", "notice"),
        "Either side can end the agreement with 30 days' written notice {nimbushost:4}. If you cancel a "
        "prepaid annual plan you get a pro-rata refund for the full months left, and your data is deleted "
        "30 days after termination {nimbushost:4}.",
    ),
]


def _question(prompt: str) -> str:
    return prompt.rsplit("Question:", 1)[-1].strip()


def _closest_sentence(text: str, question: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return max(sentences, key=lambda s: len(words & set(re.findall(r"[a-z0-9]+", s.lower())))).strip()


class ScriptedLLM:
    def __init__(self, scripts: list[Script] | None = None):
        self.scripts = DEMO_SCRIPTS if scripts is None else scripts
        self.calls: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs: Any) -> NS:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        question = _question(prompt).lower()
        sources = [(int(n), doc, int(page), text) for n, doc, page, text in _SOURCE.findall(prompt)]
        return NS(choices=[NS(message=NS(content=self._answer(question, sources)))])

    def _answer(self, question: str, sources: list[tuple[int, str, int, str]]) -> str:
        for keywords, template in self.scripts:
            if any(k in question for k in keywords):

                def cite(match: re.Match[str]) -> str:
                    prefix, page = match.group(1), int(match.group(2))
                    for n, doc, p, _ in sources:
                        if doc.startswith(prefix) and p == page:
                            return f"[{n}]"
                    return ""  # retrieval missed that page, so the demo can't cite it

                return re.sub(r"\s+\.", ".", _REF.sub(cite, template))

        if not sources:
            return "I couldn't find that in your documents."
        n, _, _, text = sources[0]
        return f'Demo mode has no scripted answer for that, so here is the closest passage: "{_closest_sentence(text, question)}" [{n}]'


class DemoLLM(ScriptedLLM):
    def __init__(self) -> None:
        super().__init__(DEMO_SCRIPTS)
