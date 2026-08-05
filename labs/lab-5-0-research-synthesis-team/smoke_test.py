"""
Smoke test for research_synthesis_team.py.

Runs the graph with the offline stub synthesizer against three topics
chosen to hit all three confidence tiers -- full coverage, partial
coverage, and a genuine miss -- so this proves the two-agent handoff,
the typed ResearchHandoff/SynthesisReport payloads, and the confidence
logic all work correctly with zero network access and zero API key
required. Swap in make_real_synthesizer() once GROQ_API_KEY is set to
exercise the actual Groq call instead.

Run: python smoke_test.py
"""

# pyrefly: ignore [missing-import]
from research_synthesis_team import ResearchHandoff, SynthesisReport, build_graph, make_stub_synthesizer


def run_case(graph, topic: str, label: str):
    print(f"\n--- {label}: '{topic}' ---")
    result = graph.invoke({"topic": topic, "log": []})
    handoff: ResearchHandoff = result["research_handoff"]
    report: SynthesisReport = result["report"]
    print(f"  sources queried:        {handoff.sources_queried}")
    print(f"  sources with no result: {handoff.sources_with_no_results}")
    print(f"  report title:           {report.title}")
    print(f"  confidence:             {report.confidence}")
    print(f"  key findings:           {len(report.key_findings)}")
    return handoff, report


if __name__ == "__main__":
    graph = build_graph(make_stub_synthesizer())

    h1, r1 = run_case(graph, "langgraph", "FULL coverage (all 3 sources hit)")
    assert isinstance(h1, ResearchHandoff)
    assert isinstance(r1, SynthesisReport)
    assert h1.sources_with_no_results == []
    assert r1.confidence == "high"
    assert r1.topic == "langgraph"

    h2, r2 = run_case(graph, "multi-agent systems", "PARTIAL coverage (internal docs miss)")
    assert h2.sources_with_no_results == ["Internal Knowledge Base"]
    assert r2.confidence == "medium"

    h3, r3 = run_case(graph, "quantum computing", "FULL miss (not indexed anywhere)")
    assert len(h3.sources_with_no_results) == 3
    assert r3.confidence == "low"
    assert r3.key_findings  # even on a full miss, the report still says something, not empty

    # Every finding in every handoff carries the query it was made under --
    # this is the point of a *typed* inter-agent message: the Synthesis
    # Agent never has to guess what the Research Agent was even asked.
    for h in (h1, h2, h3):
        for f in h.findings:
            assert f.query_used == h.topic

    print("\nAll three confidence tiers exercised, all assertions passed.")
