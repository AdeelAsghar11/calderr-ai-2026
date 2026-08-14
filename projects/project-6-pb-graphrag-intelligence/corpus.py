"""
corpus.py — 55-document corpus for Project 6-PB GraphRAG Intelligence Phase 1.

Built strictly from the fixed entity table:
- 27 relational facts (6 founded_by, 6 company located_in, 3 parent located_in, 6 part_of, 6 works_at).
- 18 descriptive facts (prior professions for 6 founders + 6 employees, specializations for 6 companies; zero graph edges).
- 10 filler/restatement paragraphs.
Enforces the structural gap: connected facts are strictly kept in separate documents, and descriptive facts contain NO graph edges.
"""

from __future__ import annotations

# The 27 Relational Paragraphs (IDs 0 to 26)
RELATIONAL_PARAGRAPHS: list[str] = [
    # 6 Founders -> Companies (founded_by)
    "Marcus Ondiek founded Ridgeline Dynamics in 2017 to pioneer autonomous system design.",
    "Sofia Petrakis founded Nimbus Water Systems to address industrial water infrastructure challenges.",
    "Aiden Kowalczyk founded Delta Forge Manufacturing to modernize structural component fabrication.",
    "Leila Farouk founded Solstice Grid Energy to deploy scalable clean energy storage.",
    "Tomas Brennan founded Kestrel Biotech to advance molecular synthesis and therapeutic research.",
    "Naomi Iwu founded Pinnacle Cargo Systems to optimize global maritime freight networks.",
    # 6 Companies -> Cities (located_in)
    "Ridgeline Dynamics is headquartered in Denver, where its engineering operations and flight test grounds are located.",
    "Nimbus Water Systems is headquartered in Rotterdam, maintaining deepwater research labs near the commercial port.",
    "Delta Forge Manufacturing is headquartered in Nairobi, operating automated industrial fabrication facilities.",
    "Solstice Grid Energy is headquartered in Osaka, anchoring its battery manufacturing and grid research hubs.",
    "Kestrel Biotech is headquartered in Denver, expanding its clinical research laboratories in the innovation corridor.",
    "Pinnacle Cargo Systems is headquartered in Rotterdam, co-locating its primary shipping dispatch operations near the harbor.",
    # 3 Parent companies -> Cities (located_in)
    "Meridian Holdings is headquartered in Denver, occupying a primary corporate executive campus.",
    "Atlas Group is headquartered in Rotterdam, managing global infrastructure and maritime investments.",
    "Crestview Partners is headquartered in Nairobi, overseeing capital allocations across industrial subsidiaries.",
    # 6 Companies -> Parent companies (part_of)
    "Ridgeline Dynamics is part of Meridian Holdings, operating under its technology and defense portfolio.",
    "Kestrel Biotech is part of Meridian Holdings, representing its life sciences and biopharmaceutical division.",
    "Nimbus Water Systems is part of Atlas Group, operating as its environmental engineering unit.",
    "Pinnacle Cargo Systems is part of Atlas Group, anchoring its international shipping and transport division.",
    "Delta Forge Manufacturing is part of Crestview Partners, representing its heavy manufacturing division.",
    "Solstice Grid Energy is part of Crestview Partners, operating as its clean energy infrastructure subsidiary.",
    # 6 Employees -> Companies (works_at)
    "Farah Deng works at Ridgeline Dynamics as a chief systems integration engineer.",
    "Victor Amaro works at Nimbus Water Systems as a principal hydrodynamics architect.",
    "Priya Chandran works at Delta Forge Manufacturing as a senior robotics automation director.",
    "Jonas Eriksson works at Solstice Grid Energy as a principal power grid systems consultant.",
    "Ana Beloso works at Kestrel Biotech as a principal medicinal chemistry researcher.",
    "Ravi Thakkar works at Pinnacle Cargo Systems as a senior logistics telematics coordinator.",
]

# The 18 Descriptive Paragraphs (IDs 27 to 44 — NO graph relationship edges)
DESCRIPTIVE_PARAGRAPHS: list[str] = [
    # 6 Founder prior professions
    "Marcus Ondiek worked as a renewable energy engineer for a decade prior to launching his venture.",
    "Sofia Petrakis worked as a marine hydrology researcher before establishing her organization.",
    "Aiden Kowalczyk worked as an industrial robotics designer prior to establishing his firm.",
    "Leila Farouk worked as a power grid operator before creating her energy startup.",
    "Tomas Brennan worked as a molecular biologist before establishing his biotechnology enterprise.",
    "Naomi Iwu worked as a port logistics specialist before founding her maritime shipping group.",
    # 6 Employee prior professions
    "Farah Deng worked as an aerospace technician before taking her current role.",
    "Victor Amaro worked as a fluid mechanics analyst prior to joining his present team.",
    "Priya Chandran worked as a heavy machinery engineer before accepting her current position.",
    "Jonas Eriksson previously worked as an electrical grid systems analyst before joining Solstice Grid Energy.",
    "Ana Beloso worked as a pharmaceutical chemist before starting her research appointment.",
    "Ravi Thakkar worked as a supply chain analyst prior to taking his current role.",
    # 6 Company specializations
    "Ridgeline Dynamics specializes in autonomous drone systems for agriculture.",
    "Nimbus Water Systems specializes in industrial water filtration technology.",
    "Delta Forge Manufacturing specializes in automated structural steel fabrication.",
    "Solstice Grid Energy specializes in high-capacity battery storage systems.",
    "Kestrel Biotech specializes in synthetic biology research for pharmaceuticals.",
    "Pinnacle Cargo Systems specializes in cross-border maritime shipping management.",
]

# The 10 Filler / Restatement Paragraphs (IDs 45 to 54)
FILLER_PARAGRAPHS: list[str] = [
    "Denver hosts several annual technology conferences attracting hardware and software engineering talent.",
    "Rotterdam remains a pivotal European trade and maritime transport hub with extensive port infrastructure.",
    "Nairobi has developed into a major industrial technology center for East African commercial growth.",
    "Osaka leads advanced research in battery cell chemistry and high-efficiency power electronics.",
    "Meridian Holdings frequently publishes quarterly corporate reports highlighting subsidiary innovation metrics.",
    "Atlas Group recently announced capital expenditure allocations for next-generation maritime fleets.",
    "Crestview Partners sponsors annual engineering fellowships across regional technical institutes.",
    "Ridgeline Dynamics recently presented new flight telemetry data at an international robotics symposium.",
    "Kestrel Biotech published an academic whitepaper on enzyme design and synthetic pathway optimization.",
    "Pinnacle Cargo Systems upgraded its digital telematics tracking network across international shipping routes.",
]

# Full 55-paragraph combined corpus
FULL_CORPUS_PARAGRAPHS: list[str] = (
    RELATIONAL_PARAGRAPHS + DESCRIPTIVE_PARAGRAPHS + FILLER_PARAGRAPHS
)

# Known entity vocabulary and types (25 unique entities)
KNOWN_ENTITIES: dict[str, str] = {
    # Founders (6)
    "Marcus Ondiek": "person",
    "Sofia Petrakis": "person",
    "Aiden Kowalczyk": "person",
    "Leila Farouk": "person",
    "Tomas Brennan": "person",
    "Naomi Iwu": "person",
    # Employees (6)
    "Farah Deng": "person",
    "Victor Amaro": "person",
    "Priya Chandran": "person",
    "Jonas Eriksson": "person",
    "Ana Beloso": "person",
    "Ravi Thakkar": "person",
    # Companies (6)
    "Ridgeline Dynamics": "company",
    "Nimbus Water Systems": "company",
    "Delta Forge Manufacturing": "company",
    "Solstice Grid Energy": "company",
    "Kestrel Biotech": "company",
    "Pinnacle Cargo Systems": "company",
    # Parent Companies (3)
    "Meridian Holdings": "company",
    "Atlas Group": "company",
    "Crestview Partners": "company",
    # Cities (4)
    "Denver": "place",
    "Rotterdam": "place",
    "Nairobi": "place",
    "Osaka": "place",
}
