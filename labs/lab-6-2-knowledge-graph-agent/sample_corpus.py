"""
sample_corpus.py — Synthetic 20-paragraph corpus for Lab 6.2 Knowledge Graph Agent.

Contains 10 core single-fact relationship paragraphs and 10 entity-restating filler paragraphs.
Crucially, connected facts are strictly kept in separate paragraphs to ensure multi-hop path traversal
is required for cross-entity queries and keyword search fails as expected.
"""

CORPUS_PARAGRAPHS: list[str] = [
    # Paragraph 0: Core Fact 1 (Dana Voss -> Trailmark Robotics)
    "Dana Voss founded Trailmark Robotics in 2018 after years of research in autonomous navigation software.",
    # Paragraph 1: Core Fact 2 (Trailmark Robotics -> Austin)
    "Trailmark Robotics is headquartered in Austin, where its primary testing facilities and engineering labs are located.",
    # Paragraph 2: Core Fact 3 (Trailmark Robotics -> Vantage Industries)
    "Trailmark Robotics is part of Vantage Industries, operating under its industrial automation technology portfolio.",
    # Paragraph 3: Core Fact 4 (Rina Achebe -> Glacier Analytics)
    "Rina Achebe founded Glacier Analytics to build enterprise data visualization tools for high-throughput streaming systems.",
    # Paragraph 4: Core Fact 5 (Glacier Analytics -> Toronto)
    "Glacier Analytics is headquartered in Toronto, maintaining a large research lab near the financial district.",
    # Paragraph 5: Core Fact 6 (Owen Kessler -> Glacier Analytics)
    "Owen Kessler works at Glacier Analytics as a principal software architect overseeing distributed database query engines.",
    # Paragraph 6: Core Fact 7 (Vantage Industries -> Seattle)
    "Vantage Industries is headquartered in Seattle, occupying a large corporate campus in the downtown commercial core.",
    # Paragraph 7: Core Fact 8 (Priya Nandan -> Cobalt Freight)
    "Priya Nandan works at Cobalt Freight as a senior logistics operations director managing international shipping pipelines.",
    # Paragraph 8: Core Fact 9 (Cobalt Freight -> Berlin)
    "Cobalt Freight is headquartered in Berlin, anchoring its European supply chain management and transport network.",
    # Paragraph 9: Core Fact 10 (Cobalt Freight -> Vantage Industries)
    "Cobalt Freight is part of Vantage Industries, representing the logistics and supply chain division of the holding conglomerate.",
    # Paragraph 10: Filler restating Dana Voss
    "Dana Voss frequently presents at international robotics conferences regarding safety standards and AI ethics.",
    # Paragraph 11: Filler restating Trailmark Robotics
    "Trailmark Robotics recently announced an expansion of its hardware testing facilities to support next-generation sensors.",
    # Paragraph 12: Filler restating Austin
    "Austin has grown into a major technological hub attracting software developers and hardware engineering talent.",
    # Paragraph 13: Filler restating Vantage Industries
    "Vantage Industries continues to report strong fiscal quarterly growth across all its enterprise subsidiaries.",
    # Paragraph 14: Filler restating Rina Achebe
    "Rina Achebe serves as a keynote speaker on distributed computing architecture and machine learning algorithms.",
    # Paragraph 15: Filler restating Glacier Analytics
    "Glacier Analytics published an open-source library for high-speed time-series data analysis.",
    # Paragraph 16: Filler restating Toronto
    "Toronto hosts several annual technology summits focusing on cloud infrastructure and data analytics.",
    # Paragraph 17: Filler restating Owen Kessler
    "Owen Kessler published several technical whitepapers on optimizing database query execution speed.",
    # Paragraph 18: Filler restating Cobalt Freight
    "Cobalt Freight recently upgraded its global tracking platform to provide real-time shipping telematics.",
    # Paragraph 19: Filler restating Priya Nandan
    "Priya Nandan leads a cross-functional engineering team building automated container management software.",
]
