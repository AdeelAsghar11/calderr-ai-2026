# 100 Wikipedia-sourced sentences across 20 categories (5 per category)
# Organized so semantic clustering is meaningful and PCA visualization is clear.

SENTENCES = [
    # --- Physics (0–4) ---
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape.",
    "The Higgs boson was discovered at CERN's Large Hadron Collider in 2012.",
    "Quantum entanglement allows two particles to share a correlated state regardless of the distance between them.",
    "General relativity describes gravity as the curvature of spacetime caused by mass and energy.",

    # --- Chemistry (5–9) ---
    "The periodic table organises all known chemical elements by their atomic number and electron configuration.",
    "Water molecules consist of two hydrogen atoms covalently bonded to a single oxygen atom.",
    "DNA is a double-helix polymer made of nucleotides that encodes genetic information.",
    "Carbon forms more compounds than any other element due to its four available bonding electrons.",
    "Alexander Fleming discovered penicillin in 1928, transforming the treatment of bacterial infections.",

    # --- Biology (10–14) ---
    "Mitochondria produce ATP through cellular respiration and are often called the powerhouse of the cell.",
    "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen in plant cells.",
    "The human genome contains approximately 3 billion base pairs of DNA encoding around 20,000 genes.",
    "Charles Darwin proposed the theory of evolution by natural selection in On the Origin of Species in 1859.",
    "CRISPR-Cas9 is a gene-editing technology that allows scientists to precisely modify DNA sequences.",

    # --- Mathematics (15–19) ---
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the other two squares.",
    "Prime numbers are integers greater than one that have no divisors other than one and themselves.",
    "Calculus was independently developed by Isaac Newton and Gottfried Wilhelm Leibniz in the 17th century.",
    "The Fibonacci sequence appears in many natural phenomena, including the spiral growth patterns of plants.",
    "Euler's identity combines five fundamental mathematical constants in a single elegant equation.",

    # --- History: Ancient (20–24) ---
    "The Great Pyramid of Giza was built around 2560 BCE as a tomb for Pharaoh Khufu.",
    "At its peak, the Roman Empire stretched from Britain in the northwest to Mesopotamia in the east.",
    "The Ancient Greeks developed the conceptual foundations of democracy in the city-state of Athens.",
    "The Silk Road was a network of trade routes connecting China to the Mediterranean for over a thousand years.",
    "Julius Caesar was assassinated on the Ides of March in 44 BCE by a group of Roman senators.",

    # --- History: Modern (25–29) ---
    "The French Revolution began in 1789 and led to the abolition of the French monarchy.",
    "World War II ended in 1945 with Germany surrendering in May and Japan surrendering in September.",
    "The Berlin Wall fell on 9 November 1989, marking the beginning of German reunification.",
    "Mahatma Gandhi led India's independence movement using non-violent civil disobedience against British rule.",
    "The Apollo 11 mission successfully landed humans on the Moon on 20 July 1969.",

    # --- Geography (30–34) ---
    "Mount Everest, located in the Himalayas, is the highest mountain on Earth at 8,849 metres above sea level.",
    "The Amazon River discharges more fresh water into the ocean than any other river on Earth.",
    "The Sahara Desert spans approximately 9.2 million square kilometres across northern Africa.",
    "The Pacific Ocean covers more than 30 percent of the Earth's total surface area.",
    "Antarctica is the coldest, driest, and windiest continent and contains about 70 percent of the world's fresh water.",

    # --- Countries & Cities (35–39) ---
    "Tokyo is the most populous metropolitan area in the world, with over 37 million inhabitants.",
    "Brazil is the largest country in South America and the fifth largest in the world by total area.",
    "The city of Venice is built on a network of 118 islands connected by over 400 bridges.",
    "India surpassed China in 2023 to become the world's most populous country.",
    "Canada is the second largest country in the world by total area, after Russia.",

    # --- Computing (40–44) ---
    "ENIAC, completed in 1945 at the University of Pennsylvania, was one of the first programmable electronic computers.",
    "The Internet evolved from ARPANET, a network originally funded by the United States Department of Defence.",
    "The Python programming language was created by Guido van Rossum and first released in 1991.",
    "Moore's Law predicted that the number of transistors on a microchip would double approximately every two years.",
    "Blockchain is a distributed ledger technology that underpins cryptocurrencies such as Bitcoin and Ethereum.",

    # --- Artificial Intelligence (45–49) ---
    "The Turing Test, proposed by Alan Turing in 1950, evaluates a machine's ability to exhibit intelligent behaviour.",
    "Deep learning uses artificial neural networks with many layers to learn hierarchical representations from data.",
    "AlphaGo became the first computer program to defeat a reigning world champion Go player in 2016.",
    "Large language models are pre-trained on massive text corpora using the transformer architecture.",
    "Reinforcement learning from human feedback is a technique used to align language models with human preferences.",

    # --- Music (50–54) ---
    "Ludwig van Beethoven composed his celebrated Ninth Symphony after becoming completely deaf.",
    "The Beatles, formed in Liverpool in 1960, became the best-selling music artists in history.",
    "Jazz music originated in New Orleans in the early 20th century, blending African rhythms with European harmonies.",
    "Wolfgang Amadeus Mozart began composing music at age five and produced over 600 works before dying at 35.",
    "Bob Dylan was awarded the Nobel Prize in Literature in 2016 for creating new poetic expressions in the American song tradition.",

    # --- Literature (55–59) ---
    "William Shakespeare wrote 37 plays and 154 sonnets, profoundly shaping the English language and literary tradition.",
    "Nineteen Eighty-Four by George Orwell depicts a totalitarian dystopia and introduced concepts such as doublethink.",
    "The Harry Potter series by J.K. Rowling has sold over 500 million copies worldwide in more than 80 languages.",
    "Gabriel García Márquez pioneered magical realism with his novel One Hundred Years of Solitude.",
    "Homer's Iliad and Odyssey are among the oldest surviving works of Western literature, composed around the 8th century BCE.",

    # --- Visual Arts (60–64) ---
    "The Mona Lisa, painted by Leonardo da Vinci, is the most visited artwork in the world and hangs in the Louvre.",
    "Vincent van Gogh created over 2,100 artworks in a decade despite suffering from severe mental illness.",
    "Pablo Picasso co-founded the Cubist movement and is considered one of the most influential artists of the 20th century.",
    "Michelangelo painted the Sistine Chapel ceiling over four years between 1508 and 1512.",
    "Photography was invented in the early 19th century and transformed how humans record and document reality.",

    # --- Sports (65–69) ---
    "The FIFA World Cup is the most widely viewed sporting event in the world, held every four years.",
    "Usain Bolt holds the world records in both the 100 metres and 200 metres sprint disciplines.",
    "The modern Olympic Games originated in ancient Greece around 776 BCE and were revived in Athens in 1896.",
    "Michael Jordan won six NBA championships with the Chicago Bulls and is widely regarded as the greatest basketball player.",
    "Cricket is the second most popular sport in the world, with a large following in South Asia, Australia, and England.",

    # --- Food & Culture (70–74) ---
    "Sushi is a Japanese dish of vinegared rice combined with seafood, vegetables, or egg, often wrapped in nori.",
    "The Mediterranean diet emphasises olive oil, vegetables, fish, and whole grains and is associated with longevity.",
    "Coffee was first cultivated in Ethiopia and has become one of the world's most traded commodities.",
    "Pizza originated in Naples, Italy, in the 18th century and has since become a globally popular food.",
    "Ramadan is the ninth month of the Islamic calendar, during which Muslims fast from dawn to sunset each day.",

    # --- Nature & Environment (75–79) ---
    "The Amazon Rainforest produces approximately 20 percent of the world's oxygen through photosynthesis.",
    "Climate change is causing global average temperatures to rise primarily due to greenhouse gas emissions.",
    "Coral reefs cover less than one percent of the ocean floor but support approximately 25 percent of all marine species.",
    "The Great Barrier Reef in Australia is the world's largest coral reef system and is visible from outer space.",
    "Deforestation destroys habitats for millions of species and is a major contributor to carbon dioxide emissions.",

    # --- Space & Astronomy (80–84) ---
    "The Milky Way galaxy is estimated to contain between 200 and 400 billion stars.",
    "Olympus Mons on Mars is the largest volcano in the solar system, roughly three times the height of Mount Everest.",
    "The James Webb Space Telescope, launched in December 2021, can observe galaxies from the earliest era of the universe.",
    "Voyager 1, launched in 1977, has travelled farther from Earth than any other human-made object.",
    "The existence of dark matter is inferred from its gravitational effects on visible matter and light.",

    # --- Medicine & Health (85–89) ---
    "The human heart beats approximately 100,000 times per day and pumps around 7,600 litres of blood.",
    "Vaccines work by training the immune system to recognise and defend against specific pathogens.",
    "The COVID-19 pandemic, caused by the SARS-CoV-2 coronavirus, began spreading globally in early 2020.",
    "Cancer is characterised by uncontrolled cell division that can invade and destroy surrounding body tissue.",
    "The human brain contains approximately 86 billion neurons connected by an estimated 100 trillion synapses.",

    # --- Economics (90–94) ---
    "Gross domestic product measures the total monetary value of goods and services produced within a country.",
    "The stock market crash of October 1929 triggered the Great Depression, the worst economic crisis of the 20th century.",
    "Supply and demand is the fundamental economic model explaining how prices are determined in a free market.",
    "Amazon was founded by Jeff Bezos in 1994 as an online bookstore before expanding into a global retail giant.",
    "Microfinance provides small loans to entrepreneurs in developing countries who lack access to traditional banking.",

    # --- Psychology & Society (95–99) ---
    "Cognitive biases are systematic patterns of thought that cause people to deviate from rational judgement.",
    "The Stanford Prison Experiment of 1971 demonstrated how institutional roles can dramatically alter human behaviour.",
    "Social media has fundamentally changed how people communicate, consume news, and form communities.",
    "Maslow's hierarchy of needs describes human motivation as a pyramid from basic physiological needs to self-actualisation.",
    "Urbanisation describes the ongoing shift of populations from rural areas to cities, reshaping global demographics.",
]

# Category metadata for visualisation (used in visualize command)
CATEGORIES = [
    "Physics", "Chemistry", "Biology", "Mathematics",
    "History (Ancient)", "History (Modern)", "Geography", "Countries & Cities",
    "Computing", "Artificial Intelligence",
    "Music", "Literature", "Visual Arts", "Sports",
    "Food & Culture", "Nature & Environment", "Space & Astronomy",
    "Medicine & Health", "Economics", "Psychology & Society",
]

assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"
assert len(CATEGORIES) == 20
