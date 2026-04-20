"""
30 adversarial test cases for QueryForge multi-model evaluation.

Each case has:
  - nl_query    : natural language input fed to the model
  - ground_truth: the exact JSON the model should output
  - category    : for analysis grouping
  - notes       : the adversarial challenge this case tests
"""

TEST_CASES = [
    # ── Category 1: Adversarial Recalls (6) ───────────────────────────────────
    {
        "id": 1,
        "category": "adversarial_recalls",
        "nl_query": "hey did toyota recall the 2020 camry or not",
        "ground_truth": {"endpoint": "recalls", "make": "TOYOTA", "model": "CAMRY", "year": "2020"},
        "notes": "Casual question with filler words; no formal recall keyword",
    },
    {
        "id": 2,
        "category": "adversarial_recalls",
        "nl_query": "toyta cmary 2020 recall",
        "ground_truth": {"endpoint": "recalls", "make": "TOYOTA", "model": "CAMRY", "year": "2020"},
        "notes": "Simultaneous typos in both make and model",
    },
    {
        "id": 3,
        "category": "adversarial_recalls",
        "nl_query": "Civic 2019 recall",
        "ground_truth": {"endpoint": "recalls", "make": "HONDA", "model": "CIVIC", "year": "2019"},
        "notes": "Make omitted; model name alone must imply HONDA",
    },
    {
        "id": 4,
        "category": "adversarial_recalls",
        "nl_query": "2018 Ford F-150 的召回記錄有哪些？",
        "ground_truth": {"endpoint": "recalls", "make": "FORD", "model": "F-150", "year": "2018"},
        "notes": "English vehicle with Chinese tail asking for recall records",
    },
    {
        "id": 5,
        "category": "adversarial_recalls",
        "nl_query": "recalls on chevy silverado from 2021",
        "ground_truth": {"endpoint": "recalls", "make": "CHEVROLET", "model": "SILVERADO", "year": "2021"},
        "notes": "Informal abbreviation chevy must expand to CHEVROLET",
    },
    {
        "id": 6,
        "category": "adversarial_recalls",
        "nl_query": "any 2022 Model 3 recall notices?",
        "ground_truth": {"endpoint": "recalls", "make": "TESLA", "model": "MODEL 3", "year": "2022"},
        "notes": "Make omitted; Model 3 uniquely implies TESLA; plural synonym notices",
    },

    # ── Category 2: Adversarial Complaints (6) ────────────────────────────────
    {
        "id": 7,
        "category": "adversarial_complaints",
        "nl_query": "people keep reporting issues with their 2017 jeep grand cherokee",
        "ground_truth": {"endpoint": "complaints", "make": "JEEP", "model": "GRAND CHEROKEE", "year": "2017"},
        "notes": "Implicit complaint intent; no complaint keyword; two-word model name",
    },
    {
        "id": 8,
        "category": "adversarial_complaints",
        "nl_query": "2018 chvrolet slverado problms",
        "ground_truth": {"endpoint": "complaints", "make": "CHEVROLET", "model": "SILVERADO", "year": "2018"},
        "notes": "Typos in make, model, and intent keyword simultaneously",
    },
    {
        "id": 9,
        "category": "adversarial_complaints",
        "nl_query": "2019 Honda Civic 消費者投訴有哪些",
        "ground_truth": {"endpoint": "complaints", "make": "HONDA", "model": "CIVIC", "year": "2019"},
        "notes": "English vehicle followed by Chinese complaint keyword 消費者投訴",
    },
    {
        "id": 10,
        "category": "adversarial_complaints",
        "nl_query": "Ford F-150 2017 transmission problems reported by owners",
        "ground_truth": {"endpoint": "complaints", "make": "FORD", "model": "F-150", "year": "2017"},
        "notes": "No complaint keyword; technical component issue must map to complaints endpoint",
    },
    {
        "id": 11,
        "category": "adversarial_complaints",
        "nl_query": "Grand Cherokee 2017 complaints",
        "ground_truth": {"endpoint": "complaints", "make": "JEEP", "model": "GRAND CHEROKEE", "year": "2017"},
        "notes": "JEEP must be inferred from two-word model name alone",
    },
    {
        "id": 12,
        "category": "adversarial_complaints",
        "nl_query": "Toyota Camry 2020 engine problems",
        "ground_truth": {"endpoint": "complaints", "make": "TOYOTA", "model": "CAMRY", "year": "2020"},
        "notes": "Engine problems is a component-level description; must route to complaints; component filter unsupported by NHTSA",
    },

    # ── Category 3: Adversarial Safety Ratings (5) ────────────────────────────
    {
        "id": 13,
        "category": "adversarial_safety_ratings",
        "nl_query": "would the 2022 Ford F-150 protect my family in a crash?",
        "ground_truth": {"endpoint": "safetyRatings", "make": "FORD", "model": "F-150", "year": "2022"},
        "notes": "Emotional implicit phrasing; crash protection intent must map to safetyRatings",
    },
    {
        "id": 14,
        "category": "adversarial_safety_ratings",
        "nl_query": "2023 Subaru Outback NCAP rating",
        "ground_truth": {"endpoint": "safetyRatings", "make": "SUBARU", "model": "OUTBACK", "year": "2023"},
        "notes": "NCAP acronym as proxy for safety ratings; not a NHTSA-specific term",
    },
    {
        "id": 15,
        "category": "adversarial_safety_ratings",
        "nl_query": "how many stars does the 2019 Honda Civic get?",
        "ground_truth": {"endpoint": "safetyRatings", "make": "HONDA", "model": "CIVIC", "year": "2019"},
        "notes": "Star rating colloquialism for the NHTSA 5-star safety program",
    },
    {
        "id": 16,
        "category": "adversarial_safety_ratings",
        "nl_query": "frontal crash test score for the 2021 Chevrolet Silverado",
        "ground_truth": {"endpoint": "safetyRatings", "make": "CHEVROLET", "model": "SILVERADO", "year": "2021"},
        "notes": "Specific crash test sub-type; should still resolve to top-level safetyRatings endpoint",
    },
    {
        "id": 17,
        "category": "adversarial_safety_ratings",
        "nl_query": "2020 Toyota RAV4 安全評等",
        "ground_truth": {"endpoint": "safetyRatings", "make": "TOYOTA", "model": "RAV4", "year": "2020"},
        "notes": "Chinese safety rating keyword 安全評等 appended to English vehicle",
    },

    # ── Category 4: Make Inference (4) ────────────────────────────────────────
    {
        "id": 18,
        "category": "make_inference",
        "nl_query": "2021 prius recalls",
        "ground_truth": {"endpoint": "recalls", "make": "TOYOTA", "model": "PRIUS", "year": "2021"},
        "notes": "Prius is uniquely associated with Toyota; make must be inferred from model alone",
    },
    {
        "id": 19,
        "category": "make_inference",
        "nl_query": "wrangler 2022 complaints",
        "ground_truth": {"endpoint": "complaints", "make": "JEEP", "model": "WRANGLER", "year": "2022"},
        "notes": "Wrangler strongly implies Jeep; lowercase; no other context",
    },
    {
        "id": 20,
        "category": "make_inference",
        "nl_query": "model s 2021 safety rating",
        "ground_truth": {"endpoint": "safetyRatings", "make": "TESLA", "model": "MODEL S", "year": "2021"},
        "notes": "Model S uniquely implies Tesla; lowercase; no make given",
    },
    {
        "id": 21,
        "category": "make_inference",
        "nl_query": "2023 mustang safety rating",
        "ground_truth": {"endpoint": "safetyRatings", "make": "FORD", "model": "MUSTANG", "year": "2023"},
        "notes": "Mustang implies Ford; tests make inference with safety ratings endpoint",
    },

    # ── Category 5: Noise and Descriptor Stripping (3) ────────────────────────
    {
        "id": 22,
        "category": "noise_and_descriptors",
        "nl_query": "2020 Toyota Camry SE AWD recalls",
        "ground_truth": {"endpoint": "recalls", "make": "TOYOTA", "model": "CAMRY", "year": "2020"},
        "notes": "Trim level SE and drivetrain AWD must be stripped; NHTSA takes base model only",
    },
    {
        "id": 23,
        "category": "noise_and_descriptors",
        "nl_query": "2021 Honda Civic Sport 2.0T Hatchback complaints",
        "ground_truth": {"endpoint": "complaints", "make": "HONDA", "model": "CIVIC", "year": "2021"},
        "notes": "Sport trim, engine spec 2.0T, and body style Hatchback are all noise to discard",
    },
    {
        "id": 24,
        "category": "noise_and_descriptors",
        "nl_query": "Toyota Camry 2020 brake recalls only",
        "ground_truth": {"endpoint": "recalls", "make": "TOYOTA", "model": "CAMRY", "year": "2020"},
        "notes": "Component qualifier brake and restriction word only must be stripped; NHTSA does not support component-level filtering",
    },

    # ── Category 6: Error Cases — Hard (6) ────────────────────────────────────
    {
        "id": 25,
        "category": "error_missing_year",
        "nl_query": "Toyota Camry recalls",
        "ground_truth": {"error": "missing_year"},
        "notes": "Make and model present but no year; must not guess or default to current year",
    },
    {
        "id": 26,
        "category": "error_missing_make",
        "nl_query": "show me recalls for 2020",
        "ground_truth": {"error": "missing_make"},
        "notes": "Year given but no make or recognizable model; must not invent a vehicle",
    },
    {
        "id": 27,
        "category": "error_out_of_scope",
        "nl_query": "Which SUVs had the most recalls in 2021?",
        "ground_truth": {"error": "out_of_scope"},
        "notes": "Aggregate ranking query; NHTSA API requires specific make and model — cannot answer comparative questions",
    },
    {
        "id": 28,
        "category": "error_out_of_scope",
        "nl_query": "Best and worst safety rated car in 2022",
        "ground_truth": {"error": "out_of_scope"},
        "notes": "High-severity comparative query; no specific vehicle; would require iterating all makes and models",
    },
    {
        "id": 29,
        "category": "error_out_of_scope",
        "nl_query": "how do I reset the oil life on my 2019 Honda Civic?",
        "ground_truth": {"error": "out_of_scope"},
        "notes": "Car maintenance question with a real vehicle; vehicle presence is a trap; not a NHTSA safety data query",
    },
    {
        "id": 30,
        "category": "error_missing_year",
        "nl_query": "What are the issues with the Accord?",
        "ground_truth": {"error": "missing_year"},
        "notes": "Make can be inferred (Honda), endpoint from issues (complaints), but year is absent; must not fabricate a year",
    },
]
