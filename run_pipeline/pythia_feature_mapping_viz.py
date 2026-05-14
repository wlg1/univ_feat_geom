import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "umap-learn is required for this tool. Install with: pip install umap-learn"
    ) from exc

try:
    from run_pipeline.alignment_fns import (
        apply_mapping_method,
        build_activation_corr_matrix,
        compute_alignment_and_metrics,
        compute_subset_metrics,
    )
    from run_pipeline.get_actv_fns import get_sae_actvs
    from run_pipeline.interpret_fns import highest_activating_tokens, store_top_toks
except ImportError:
    from alignment_fns import (
        apply_mapping_method,
        build_activation_corr_matrix,
        compute_alignment_and_metrics,
        compute_subset_metrics,
    )
    from get_actv_fns import get_sae_actvs
    from interpret_fns import highest_activating_tokens, store_top_toks


# Default semantic buckets: a feature belongs to a group if its top-token label string
# (comma-separated) matches any keyword as a substring (case-insensitive), OR matches the
# optional regex "pattern". Override with --semantic-categories-json for custom groups.
DEFAULT_SEMANTIC_CATEGORIES: List[Dict[str, Any]] = [
    {
        "id": "numerical",
        "name": "Numerical & quantities",
        "keywords": [
            "number",
            "numbers",
            "numeric",
            "numeral",
            "digit",
            "digits",
            "count",
            "counting",
            "sum",
            "total",
            "subtotal",
            "average",
            "mean",
            "median",
            "mode",
            "percent",
            "percentage",
            "ratio",
            "fraction",
            "decimal",
            "integer",
            "float",
            "ordinal",
            "magnitude",
            "quantity",
            "amount",
            "measure",
            "measurement",
            "unit",
            "units",
            "scale",
            "range",
            "score",
            "rate",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "hundred",
            "thousand",
            "million",
            "billion",
            "trillion",
            "first",
            "second",
            "third",
            "fourth",
            "half",
            "double",
            "triple",
            "dozen",
            "pair",
            "single",
            "math",
            "algebra",
            "calculate",
            "calculation",
            "equation",
            "equal",
            "plus",
            "minus",
            "times",
            "multiply",
            "divide",
            "divisor",
            "remainder",
            "greater",
            "less",
            "more",
            "fewer",
            "approx",
            "infinity",
        ],
        "pattern": r"\d",
    },
    {
        "id": "time_calendar",
        "name": "Time & calendar",
        "keywords": [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "day",
            "week",
            "month",
            "year",
            "hour",
            "minute",
            "second",
            "moment",
            "time",
            "clock",
            "calendar",
            "schedule",
            "morning",
            "afternoon",
            "evening",
            "night",
            "midnight",
            "noon",
            "today",
            "tomorrow",
            "yesterday",
            "season",
            "winter",
            "spring",
            "summer",
            "autumn",
            "fall",
            "decade",
            "century",
            "era",
            "deadline",
            "duration",
            "interval",
            "delay",
            "early",
            "late",
            "ago",
            "since",
            "until",
        ],
    },
    {
        "id": "emotions",
        "name": "Emotions & affect",
        "keywords": [
            "happy",
            "happiness",
            "sad",
            "sadness",
            "angry",
            "anger",
            "fear",
            "afraid",
            "scared",
            "love",
            "hate",
            "joy",
            "cry",
            "crying",
            "smile",
            "worried",
            "worry",
            "excited",
            "lonely",
            "proud",
            "shame",
            "rage",
            "emotion",
            "emotional",
            "mood",
            "feel",
            "feeling",
            "grief",
            "hope",
            "hopeless",
            "disgust",
            "surprise",
            "calm",
            "anxiety",
            "stress",
            "relief",
            "grateful",
            "jealous",
            "envy",
            "guilt",
            "regret",
        ],
    },
    {
        "id": "people_roles",
        "name": "People & social roles",
        "keywords": [
            "woman",
            "man",
            "child",
            "children",
            "boy",
            "girl",
            "baby",
            "teen",
            "adult",
            "elder",
            "king",
            "queen",
            "prince",
            "princess",
            "teacher",
            "doctor",
            "nurse",
            "patient",
            "friend",
            "enemy",
            "neighbor",
            "mother",
            "father",
            "parent",
            "brother",
            "sister",
            "family",
            "people",
            "person",
            "human",
            "crowd",
            "citizen",
            "leader",
            "boss",
            "worker",
            "student",
            "soldier",
            "officer",
            "police",
            "judge",
            "lawyer",
            "author",
            "artist",
            "actor",
            "celebrity",
        ],
    },
    {
        "id": "animals",
        "name": "Animals",
        "keywords": [
            "animal",
            "dog",
            "cat",
            "bird",
            "fish",
            "horse",
            "cow",
            "pig",
            "sheep",
            "goat",
            "chicken",
            "mouse",
            "rat",
            "snake",
            "bear",
            "wolf",
            "lion",
            "tiger",
            "elephant",
            "monkey",
            "whale",
            "shark",
            "insect",
            "bee",
            "spider",
            "butterfly",
            "pet",
            "wildlife",
            "species",
            "mammal",
            "reptile",
            "paw",
            "tail",
            "nest",
            "egg",
            "feather",
        ],
    },
    {
        "id": "plants_nature",
        "name": "Plants, trees & land",
        "keywords": [
            "tree",
            "forest",
            "wood",
            "leaf",
            "flower",
            "grass",
            "plant",
            "garden",
            "seed",
            "root",
            "branch",
            "bush",
            "vine",
            "crop",
            "soil",
            "rock",
            "stone",
            "mountain",
            "valley",
            "river",
            "lake",
            "ocean",
            "island",
            "desert",
            "jungle",
            "meadow",
            "field",
            "earth",
            "ground",
            "hill",
            "cave",
            "shore",
            "wave",
            "tide",
        ],
    },
    {
        "id": "food_drink",
        "name": "Food & drink",
        "keywords": [
            "food",
            "eat",
            "meal",
            "breakfast",
            "lunch",
            "dinner",
            "snack",
            "bread",
            "meat",
            "fish",
            "fruit",
            "vegetable",
            "rice",
            "soup",
            "salad",
            "cheese",
            "milk",
            "water",
            "drink",
            "coffee",
            "tea",
            "wine",
            "beer",
            "juice",
            "sugar",
            "salt",
            "cook",
            "recipe",
            "kitchen",
            "restaurant",
            "hungry",
            "thirsty",
            "sweet",
            "bitter",
            "spice",
        ],
    },
    {
        "id": "sports_games",
        "name": "Sports & games",
        "keywords": [
            "sport",
            "game",
            "play",
            "player",
            "team",
            "score",
            "win",
            "lose",
            "match",
            "ball",
            "goal",
            "race",
            "run",
            "jump",
            "swim",
            "fight",
            "boxing",
            "soccer",
            "football",
            "basketball",
            "baseball",
            "tennis",
            "golf",
            "chess",
            "card",
            "deck",
            "dice",
            "toy",
            "puzzle",
            "olympic",
            "stadium",
            "coach",
            "referee",
            "tournament",
            "champion",
        ],
    },
    {
        "id": "art_music",
        "name": "Art, music & culture",
        "keywords": [
            "art",
            "artist",
            "paint",
            "painting",
            "draw",
            "drawing",
            "sculpture",
            "museum",
            "gallery",
            "music",
            "song",
            "sing",
            "singer",
            "band",
            "concert",
            "guitar",
            "piano",
            "violin",
            "drum",
            "melody",
            "rhythm",
            "dance",
            "dancer",
            "theater",
            "theatre",
            "film",
            "movie",
            "cinema",
            "novel",
            "poem",
            "poetry",
            "story",
            "fiction",
            "drama",
            "comedy",
            "photo",
            "camera",
        ],
    },
    {
        "id": "geography_places",
        "name": "Geography & places",
        "keywords": [
            "city",
            "town",
            "village",
            "country",
            "nation",
            "state",
            "capital",
            "region",
            "border",
            "map",
            "street",
            "road",
            "bridge",
            "airport",
            "station",
            "hotel",
            "park",
            "square",
            "building",
            "address",
            "north",
            "south",
            "east",
            "west",
            "local",
            "global",
            "abroad",
            "travel",
            "continent",
            "europe",
            "asia",
            "africa",
            "america",
            "oceania",
            "antarctic",
            "latitude",
            "longitude",
        ],
    },
    {
        "id": "body_health",
        "name": "Body & health",
        "keywords": [
            "body",
            "head",
            "brain",
            "heart",
            "blood",
            "bone",
            "muscle",
            "skin",
            "eye",
            "ear",
            "hand",
            "foot",
            "leg",
            "arm",
            "pain",
            "hurt",
            "sick",
            "illness",
            "disease",
            "health",
            "healthy",
            "medicine",
            "drug",
            "hospital",
            "surgery",
            "virus",
            "bacteria",
            "cell",
            "gene",
            "dna",
            "sleep",
            "wake",
            "breath",
            "diet",
            "exercise",
            "weight",
            "fever",
            "cough",
            "wound",
        ],
    },
    {
        "id": "business_money",
        "name": "Business & money",
        "keywords": [
            "money",
            "dollar",
            "euro",
            "price",
            "cost",
            "pay",
            "payment",
            "buy",
            "sell",
            "sale",
            "market",
            "stock",
            "trade",
            "profit",
            "loss",
            "debt",
            "loan",
            "bank",
            "tax",
            "budget",
            "salary",
            "wage",
            "rich",
            "poor",
            "economy",
            "business",
            "company",
            "corporate",
            "invest",
            "contract",
            "invoice",
            "revenue",
            "expense",
            "account",
            "finance",
            "crypto",
            "bitcoin",
        ],
    },
    {
        "id": "education",
        "name": "Education & science",
        "keywords": [
            "school",
            "university",
            "college",
            "student",
            "teacher",
            "class",
            "course",
            "lesson",
            "homework",
            "exam",
            "test",
            "grade",
            "degree",
            "learn",
            "study",
            "research",
            "paper",
            "theory",
            "experiment",
            "lab",
            "science",
            "physics",
            "chemistry",
            "biology",
            "atom",
            "molecule",
            "energy",
            "force",
            "mass",
            "speed",
            "light",
            "space",
            "planet",
            "star",
            "galaxy",
            "evolution",
            "climate",
        ],
    },
    {
        "id": "transport",
        "name": "Transport & vehicles",
        "keywords": [
            "car",
            "bus",
            "train",
            "plane",
            "ship",
            "boat",
            "bike",
            "bicycle",
            "motor",
            "engine",
            "wheel",
            "road",
            "traffic",
            "drive",
            "driver",
            "passenger",
            "fuel",
            "gas",
            "electric",
            "vehicle",
            "truck",
            "taxi",
            "subway",
            "rail",
            "flight",
            "airline",
            "pilot",
            "harbor",
            "dock",
            "speed",
            "mile",
            "kilometer",
        ],
    },
    {
        "id": "technology_code",
        "name": "Technology & computing",
        "keywords": [
            "computer",
            "software",
            "hardware",
            "code",
            "coding",
            "program",
            "algorithm",
            "data",
            "database",
            "server",
            "network",
            "internet",
            "website",
            "email",
            "file",
            "folder",
            "memory",
            "cpu",
            "gpu",
            "cloud",
            "api",
            "bug",
            "debug",
            "compile",
            "script",
            "python",
            "java",
            "javascript",
            "html",
            "css",
            "sql",
            "linux",
            "windows",
            "keyboard",
            "screen",
            "pixel",
            "digital",
            "binary",
            "encrypt",
            "password",
        ],
    },
    {
        "id": "law_government",
        "name": "Law & government",
        "keywords": [
            "law",
            "legal",
            "court",
            "trial",
            "judge",
            "jury",
            "crime",
            "criminal",
            "police",
            "prison",
            "sentence",
            "rights",
            "constitution",
            "government",
            "state",
            "parliament",
            "congress",
            "senate",
            "president",
            "minister",
            "vote",
            "election",
            "policy",
            "tax",
            "regulation",
            "license",
            "contract",
            "lawsuit",
            "evidence",
            "witness",
            "guilty",
            "innocent",
        ],
    },
    {
        "id": "religion_philosophy",
        "name": "Religion & philosophy",
        "keywords": [
            "god",
            "gods",
            "religion",
            "faith",
            "prayer",
            "church",
            "temple",
            "mosque",
            "bible",
            "soul",
            "spirit",
            "heaven",
            "hell",
            "sin",
            "sacred",
            "holy",
            "worship",
            "priest",
            "monk",
            "philosophy",
            "ethics",
            "moral",
            "truth",
            "existence",
            "meaning",
            "logic",
            "reason",
            "belief",
            "atheist",
            "ritual",
            "meditation",
            "karma",
            "buddha",
            "christ",
            "islam",
            "jewish",
            "hindu",
        ],
    },
    {
        "id": "household_objects",
        "name": "Household & objects",
        "keywords": [
            "house",
            "home",
            "room",
            "door",
            "window",
            "wall",
            "floor",
            "ceiling",
            "roof",
            "bed",
            "chair",
            "table",
            "lamp",
            "light",
            "mirror",
            "box",
            "bag",
            "bottle",
            "cup",
            "plate",
            "knife",
            "fork",
            "spoon",
            "tool",
            "hammer",
            "key",
            "lock",
            "paper",
            "book",
            "shelf",
            "closet",
            "towel",
            "soap",
            "brush",
            "clock",
            "phone",
            "television",
            "radio",
        ],
    },
    {
        "id": "weather",
        "name": "Weather & atmosphere",
        "keywords": [
            "weather",
            "rain",
            "snow",
            "storm",
            "wind",
            "cloud",
            "sun",
            "sky",
            "fog",
            "thunder",
            "lightning",
            "hurricane",
            "tornado",
            "flood",
            "drought",
            "temperature",
            "humid",
            "dry",
            "cold",
            "hot",
            "warm",
            "freeze",
            "melt",
            "forecast",
            "climate",
            "atmosphere",
            "pressure",
            "breeze",
            "hail",
            "sleet",
            "icicle",
            "rainbow",
        ],
    },
    {
        "id": "colors_shapes",
        "name": "Colors & shapes",
        "keywords": [
            "color",
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "gray",
            "grey",
            "orange",
            "purple",
            "pink",
            "brown",
            "gold",
            "silver",
            "dark",
            "bright",
            "shade",
            "shape",
            "circle",
            "square",
            "triangle",
            "line",
            "curve",
            "angle",
            "corner",
            "flat",
            "round",
            "straight",
            "symmetry",
            "pattern",
            "texture",
            "transparent",
            "opaque",
        ],
    },
    {
        "id": "communication",
        "name": "Communication & language",
        "keywords": [
            "word",
            "words",
            "sentence",
            "language",
            "speak",
            "say",
            "tell",
            "talk",
            "conversation",
            "message",
            "letter",
            "write",
            "read",
            "text",
            "speech",
            "voice",
            "listen",
            "hear",
            "call",
            "answer",
            "question",
            "reply",
            "meaning",
            "translate",
            "grammar",
            "syntax",
            "phrase",
            "paragraph",
            "quote",
            "mention",
            "announce",
            "whisper",
            "shout",
            "silence",
            "noise",
        ],
    },
    {
        "id": "negation_logic",
        "name": "Negation & logic",
        "keywords": [
            "not",
            "no",
            "never",
            "none",
            "nothing",
            "nobody",
            "nowhere",
            "neither",
            "nor",
            "without",
            "lack",
            "absent",
            "deny",
            "refuse",
            "reject",
            "false",
            "true",
            "maybe",
            "perhaps",
            "if",
            "then",
            "else",
            "because",
            "therefore",
            "however",
            "although",
            "unless",
            "whether",
            "either",
            "both",
            "all",
            "any",
            "every",
            "some",
            "such",
            "same",
            "different",
            "versus",
        ],
    },
    {
        "id": "questions_uncertainty",
        "name": "Questions & uncertainty",
        "keywords": [
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "where",
            "when",
            "why",
            "how",
            "whether",
            "could",
            "would",
            "should",
            "might",
            "may",
            "can",
            "must",
            "uncertain",
            "unclear",
            "unknown",
            "guess",
            "assume",
            "probably",
            "possibly",
            "likely",
            "unlikely",
            "doubt",
            "sure",
            "certain",
            "confuse",
            "wonder",
            "ask",
        ],
    },
    {
        "id": "size_space",
        "name": "Size, space & distance",
        "keywords": [
            "big",
            "small",
            "large",
            "tiny",
            "huge",
            "wide",
            "narrow",
            "tall",
            "short",
            "long",
            "deep",
            "shallow",
            "high",
            "low",
            "near",
            "far",
            "distance",
            "close",
            "away",
            "inside",
            "outside",
            "between",
            "among",
            "around",
            "across",
            "beyond",
            "above",
            "below",
            "beside",
            "center",
            "edge",
            "middle",
            "corner",
            "volume",
            "area",
            "dimension",
        ],
    },
    {
        "id": "work_labor",
        "name": "Work & labor",
        "keywords": [
            "work",
            "job",
            "labor",
            "task",
            "project",
            "office",
            "factory",
            "machine",
            "shift",
            "hire",
            "fire",
            "resume",
            "career",
            "profession",
            "skill",
            "effort",
            "busy",
            "idle",
            "break",
            "overtime",
            "union",
            "wage",
            "employ",
            "employer",
            "employee",
            "colleague",
            "meeting",
            "report",
            "email",
            "spreadsheet",
        ],
    },
    {
        "id": "military_conflict",
        "name": "Military & conflict",
        "keywords": [
            "war",
            "battle",
            "fight",
            "weapon",
            "gun",
            "bomb",
            "missile",
            "army",
            "navy",
            "air force",
            "soldier",
            "tank",
            "attack",
            "defend",
            "enemy",
            "ally",
            "peace",
            "treaty",
            "invasion",
            "siege",
            "casualty",
            "terror",
            "violence",
            "conflict",
            "strategy",
            "tactics",
            "command",
            "officer",
            "rank",
            "uniform",
            "camp",
            "fort",
            "border",
        ],
    },
    {
        "id": "internet_media",
        "name": "Internet & media",
        "keywords": [
            "internet",
            "online",
            "website",
            "web",
            "link",
            "click",
            "post",
            "blog",
            "forum",
            "social",
            "twitter",
            "facebook",
            "youtube",
            "video",
            "stream",
            "download",
            "upload",
            "meme",
            "news",
            "journalist",
            "article",
            "headline",
            "editor",
            "publish",
            "subscribe",
            "channel",
            "podcast",
            "wiki",
            "search",
            "browser",
            "cookie",
            "spam",
            "troll",
            "viral",
        ],
    },
    {
        "id": "death_risk",
        "name": "Death, risk & harm",
        "keywords": [
            "death",
            "die",
            "dead",
            "kill",
            "murder",
            "suicide",
            "grave",
            "funeral",
            "risk",
            "danger",
            "harm",
            "hurt",
            "injury",
            "accident",
            "disaster",
            "emergency",
            "warning",
            "threat",
            "violence",
            "abuse",
            "poison",
            "toxic",
            "fatal",
            "survive",
            "loss",
            "grief",
            "orphan",
            "coffin",
            "bury",
        ],
    },
    {
        "id": "clothing",
        "name": "Clothing & appearance",
        "keywords": [
            "clothes",
            "shirt",
            "pants",
            "dress",
            "skirt",
            "coat",
            "jacket",
            "shoe",
            "boot",
            "hat",
            "cap",
            "glove",
            "sock",
            "belt",
            "uniform",
            "fabric",
            "cotton",
            "wool",
            "silk",
            "leather",
            "jewelry",
            "ring",
            "necklace",
            "watch",
            "glasses",
            "hair",
            "beard",
            "makeup",
            "fashion",
            "style",
            "wear",
            "naked",
            "outfit",
        ],
    },
    {
        "id": "tools_materials",
        "name": "Tools & materials",
        "keywords": [
            "tool",
            "machine",
            "metal",
            "wood",
            "plastic",
            "glass",
            "steel",
            "iron",
            "copper",
            "wire",
            "nail",
            "screw",
            "glue",
            "tape",
            "rope",
            "chain",
            "electric",
            "battery",
            "motor",
            "gear",
            "pipe",
            "concrete",
            "brick",
            "cement",
            "fabric",
            "rubber",
            "chemical",
            "oil",
            "gas",
            "paint",
            "brush",
        ],
    },
    {
        "id": "energy_power",
        "name": "Energy & power",
        "keywords": [
            "energy",
            "power",
            "electric",
            "electricity",
            "battery",
            "solar",
            "nuclear",
            "fuel",
            "oil",
            "gas",
            "coal",
            "wind",
            "turbine",
            "engine",
            "motor",
            "heat",
            "cold",
            "temperature",
            "watt",
            "volt",
            "current",
            "charge",
            "magnet",
            "radiation",
            "laser",
            "beam",
            "fire",
            "flame",
            "burn",
            "explode",
            "fusion",
            "fission",
        ],
    },
    {
        "id": "politics_society",
        "name": "Politics & society",
        "keywords": [
            "politics",
            "political",
            "party",
            "democrat",
            "republican",
            "liberal",
            "conservative",
            "vote",
            "campaign",
            "debate",
            "protest",
            "rights",
            "freedom",
            "justice",
            "equality",
            "racism",
            "sexism",
            "gender",
            "minority",
            "immigrant",
            "refugee",
            "citizen",
            "society",
            "culture",
            "tradition",
            "norm",
            "community",
            "public",
            "private",
            "social",
            "media",
            "opinion",
            "poll",
        ],
    },
    {
        "id": "abstract_concepts",
        "name": "Abstract concepts",
        "keywords": [
            "idea",
            "concept",
            "thought",
            "mind",
            "memory",
            "dream",
            "hope",
            "goal",
            "purpose",
            "value",
            "virtue",
            "beauty",
            "truth",
            "freedom",
            "justice",
            "wisdom",
            "knowledge",
            "ignorance",
            "chance",
            "luck",
            "fate",
            "destiny",
            "soul",
            "spirit",
            "essence",
            "form",
            "structure",
            "system",
            "pattern",
            "model",
            "example",
            "instance",
            "category",
            "property",
            "relation",
            "context",
        ],
    },
]


def _load_semantic_categories(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        out: List[Dict[str, Any]] = []
        for c in DEFAULT_SEMANTIC_CATEGORIES:
            entry: Dict[str, Any] = {"id": c["id"], "name": c["name"], "keywords": list(c["keywords"])}
            pat = c.get("pattern")
            if pat:
                entry["pattern"] = re.compile(str(pat), re.IGNORECASE)
            out.append(entry)
        return out
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("semantic categories JSON must be a list of objects with id, name, keywords")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"semantic category entry {i} must be an object")
        if "id" not in item or "name" not in item:
            raise ValueError(f"semantic category entry {i} needs id and name")
        kws = item.get("keywords") or item.get("keyword")
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, list) or not kws:
            raise ValueError(f"semantic category {item['id']!r} needs a non-empty keywords list")
        pat = item.get("pattern")
        entry: Dict[str, Any] = {"id": str(item["id"]), "name": str(item["name"]), "keywords": [str(k) for k in kws]}
        if pat:
            entry["pattern"] = re.compile(str(pat), re.IGNORECASE)
        out.append(entry)
    return out


def _label_matches_semantic_category(label: str, cat: Dict[str, Any]) -> bool:
    if not label or label == "<no top tokens>":
        return False
    low = label.lower()
    pat = cat.get("pattern")
    if pat is not None and pat.search(label):
        return True
    for kw in cat["keywords"]:
        if kw.lower() in low:
            return True
    return False


def _semantic_groups_for_points(
    points_a: List[Dict], points_b: List[Dict], categories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for cat in categories:
        fa = [int(p["feature_id"]) for p in points_a if _label_matches_semantic_category(str(p.get("label", "")), cat)]
        fb = [int(p["feature_id"]) for p in points_b if _label_matches_semantic_category(str(p.get("label", "")), cat)]
        groups.append({"id": cat["id"], "name": cat["name"], "feature_ids_a": fa, "feature_ids_b": fb})
    return groups


def _pair_arrays_for_semantic_group(
    g: Dict[str, Any],
    b_to_a_map: Dict[int, int],
    a_to_b_map: Dict[int, int],
    selected_b: np.ndarray,
    selected_a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unique (fa, fb) pairs for metrics: B in group with map, plus A in group with consistent reverse map."""
    sel_b = {int(x) for x in selected_b.tolist()}
    sel_a = {int(x) for x in selected_a.tolist()}
    pairs: List[Tuple[int, int]] = []
    seen: set = set()
    for fb in g.get("feature_ids_b") or []:
        fb = int(fb)
        if fb not in sel_b or fb not in b_to_a_map:
            continue
        fa = int(b_to_a_map[fb])
        key = (fa, fb)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    for fa in g.get("feature_ids_a") or []:
        fa = int(fa)
        if fa not in sel_a:
            continue
        fb = a_to_b_map.get(fa)
        if fb is None or int(fb) not in sel_b:
            continue
        fb = int(fb)
        if int(b_to_a_map.get(fb, -1)) != fa:
            continue
        key = (fa, fb)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    if not pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    fa_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    fb_arr = np.array([p[1] for p in pairs], dtype=np.int64)
    return fa_arr, fb_arr


def _load_text_batch(dataset_name: str, split: str, text_key: str, num_samples: int) -> List[str]:
    ds = load_dataset(dataset_name, split=split)
    text_rows = [row[text_key] for row in ds if row.get(text_key)]
    if len(text_rows) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but only found {len(text_rows)} rows in {dataset_name}:{split}")
    return text_rows[:num_samples]


def _get_inputs(tokenizer, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded


def _load_model(model_name: str, device: str):
    dtype = torch.float16 if (device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def _feature_pool_from_activity(reshaped: torch.Tensor, pool_size: int) -> np.ndarray:
    max_vals = reshaped.abs().max(dim=0).values.cpu().numpy()
    pool = np.argsort(max_vals)[-pool_size:]
    return np.sort(pool)


def _select_feature_sets(pool_b: np.ndarray, b_scores: np.ndarray, per_side_limit: int) -> np.ndarray:
    order = np.argsort(b_scores)
    keep = pool_b[order[-per_side_limit:]]
    return np.sort(keep)


def _top_token_label(
    feature_acts_3d: torch.Tensor,
    feature_idx: int,
    batch_tokens: torch.Tensor,
    tokenizer,
    sample_k: int,
) -> str:
    top_idx = highest_activating_tokens(
        feature_acts_3d,
        feature_idx=feature_idx,
        k=sample_k,
        batch_tokens=batch_tokens,
    )
    decoded = store_top_toks(top_idx, batch_tokens, tokenizer)
    cleaned = []
    seen = set()
    for tok in decoded:
        token_text = tok.replace("\n", "\\n").strip()
        if not token_text:
            continue
        if token_text not in seen:
            cleaned.append(token_text)
            seen.add(token_text)
    if not cleaned:
        return "<no top tokens>"
    return ", ".join(cleaned[:sample_k])


def _compute_umap(weights: np.ndarray, seed: int) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="euclidean", random_state=seed)
    return reducer.fit_transform(weights)


def _build_payload(
    model_a_name: str,
    model_b_name: str,
    sae_a_name: str,
    sae_b_name: str,
    layer_a: int,
    layer_b: int,
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b_to_a_map: Dict[int, int],
    b_to_a_corr: Dict[int, float],
    a_to_b_map: Dict[int, int],
    labels_a: Dict[int, str],
    labels_b: Dict[int, str],
    semantic_categories: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict:
    selected_a_set = set(selected_a.tolist())
    selected_b_set = set(selected_b.tolist())

    reverse_candidates: Dict[int, List[Tuple[int, float]]] = {}
    for feat_b in selected_b.tolist():
        feat_a = b_to_a_map.get(int(feat_b))
        if feat_a is None:
            continue
        reverse_candidates.setdefault(int(feat_a), []).append((int(feat_b), float(b_to_a_corr.get(int(feat_b), np.nan))))

    points_a = []
    for i, feat_a in enumerate(selected_a.tolist()):
        mapped_b = a_to_b_map.get(int(feat_a))
        if mapped_b is None:
            mapped_b = -1
        if mapped_b not in selected_b_set and feat_a in reverse_candidates:
            mapped_b = sorted(reverse_candidates[feat_a], key=lambda x: x[1], reverse=True)[0][0]
        points_a.append(
            {
                "feature_id": int(feat_a),
                "x": float(emb_a[i, 0]),
                "y": float(emb_a[i, 1]),
                "label": labels_a.get(int(feat_a), ""),
                "mapped_feature_id": int(mapped_b) if mapped_b in selected_b_set else None,
            }
        )

    points_b = []
    for i, feat_b in enumerate(selected_b.tolist()):
        feat_a = b_to_a_map.get(int(feat_b))
        points_b.append(
            {
                "feature_id": int(feat_b),
                "x": float(emb_b[i, 0]),
                "y": float(emb_b[i, 1]),
                "label": labels_b.get(int(feat_b), ""),
                "mapped_feature_id": feat_a if (feat_a in selected_a_set) else None,
                "corr_to_mapped": float(b_to_a_corr.get(int(feat_b), np.nan)),
            }
        )

    semantic_groups = _semantic_groups_for_points(points_a, points_b, semantic_categories)

    meta: Dict[str, Any] = {
        "description": "Cross-model SAE feature UMAP with activation-based alignment (greedy / Hungarian / OT)",
        "mapping_direction_note": "model_b feature -> aligned model_a feature (see mapping_method in metrics)",
    }
    if metrics is not None:
        meta["metrics"] = metrics

    return {
        "meta": meta,
        "semantic_groups": semantic_groups,
        "model_a": {
            "name": model_a_name,
            "sae_name": sae_a_name,
            "layer": layer_a,
            "points": points_a,
        },
        "model_b": {
            "name": model_b_name,
            "sae_name": sae_b_name,
            "layer": layer_b,
            "points": points_b,
        },
    }


def _render_html(payload: Dict, output_html: Path, title: str) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    payload_js = json.dumps(payload)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0d1117; color: #e6edf3; }}
    .container {{ padding: 12px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .panel {{ border: 1px solid #30363d; border-radius: 8px; padding: 8px; background: #161b22; }}
    .title {{ font-size: 14px; margin: 0 0 8px 0; color: #7ee787; }}
    .plot-panel {{ position: relative; width: 100%; height: 520px; }}
    .plot-panel .plot {{ width: 100%; height: 100%; }}
    .paint-select-canvas {{
      position: absolute; left: 0; top: 0; width: 100%; height: 100%;
      z-index: 4; pointer-events: none; touch-action: none;
    }}
    .details {{ margin-top: 8px; font-size: 12px; line-height: 1.5; white-space: pre-wrap; min-height: 90px; }}
    .toolbar {{ margin-bottom: 12px; }}
    .toolbar label {{ font-size: 12px; margin-right: 6px; color: #8b949e; }}
    .toolbar select {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px; min-width: 220px; }}
    .toolbar-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    button#reset-pins-btn {{
      background: #21262d; color: #e6edf3; border: 1px solid #484f58; border-radius: 6px;
      padding: 6px 12px; font-size: 12px; cursor: pointer;
    }}
    button#reset-pins-btn:hover {{ background: #30363d; border-color: #8b949e; }}
    label.toggle-mesh {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #c9d1d9; cursor: pointer; user-select: none; }}
    label.toggle-mesh input {{ cursor: pointer; }}
    .drag-mode-row .hdr {{ font-size: 12px; color: #8b949e; margin-right: 4px; }}
    .cat-table-wrap {{ margin-top: 14px; }}
    .cat-table-wrap .tbl-title {{ font-size: 13px; color: #7ee787; margin: 0 0 8px 0; }}
    .cat-table-scroll {{ overflow: auto; max-height: 360px; border: 1px solid #30363d; border-radius: 8px; background: #161b22; }}
    table.semantic-pairs {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    table.semantic-pairs th, table.semantic-pairs td {{ border-bottom: 1px solid #30363d; padding: 6px 10px; text-align: left; vertical-align: top; }}
    table.semantic-pairs th {{ background: #21262d; color: #8b949e; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
    table.semantic-pairs tr.pair-row {{ cursor: pointer; }}
    table.semantic-pairs tr.pair-row:not(.pair-row-selected):hover td {{ background: #1c2128; }}
    table.semantic-pairs tr.pair-row-selected td {{
      background: #1f3d56;
      box-shadow: inset 0 0 0 1px #58a6ff;
    }}
    table.semantic-pairs td.num {{ white-space: nowrap; color: #79c0ff; font-variant-numeric: tabular-nums; }}
    table.semantic-pairs td.corr {{ white-space: nowrap; color: #ffa657; }}
    .metrics-summary {{ font-size: 12px; margin: 0 0 12px 0; padding: 10px 12px; border: 1px solid #30363d; border-radius: 8px; background: #161b22; }}
    .metrics-summary h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #7ee787; }}
    .metrics-summary table {{ border-collapse: collapse; width: 100%; max-width: 920px; }}
    .metrics-summary td {{ padding: 4px 8px; border-bottom: 1px solid #21262d; vertical-align: top; }}
    .metrics-summary td.k {{ color: #8b949e; width: 46%; }}
    .metrics-summary td.v {{ color: #e6edf3; font-variant-numeric: tabular-nums; }}
    .category-metrics-panel {{ margin-top: 0; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <h3 style="margin:0 0 10px 0;">{title}</h3>
    <div id="metrics-summary" class="metrics-summary" aria-label="Feature space similarity metrics"></div>
    <div class="toolbar toolbar-row">
      <label for="category-select">Semantic group</label>
      <select id="category-select" aria-label="Highlight features by semantic group"></select>
      <button type="button" id="reset-pins-btn">Reset all selected features</button>
      <label class="toggle-mesh"><input type="checkbox" id="toggle-mesh-pinned" /> Subset mesh (selected features — pins + box/paint region)</label>
      <label class="toggle-mesh"><input type="checkbox" id="toggle-mesh-category" /> Subset mesh (category)</label>
      <span id="mesh-hint" style="font-size:11px;color:#8b949e;max-width:420px;"></span>
    </div>
    <div id="category-metrics" class="metrics-summary category-metrics-panel" style="display:none;" aria-live="polite" aria-label="Metrics for selected semantic category"></div>
    <div class="toolbar toolbar-row drag-mode-row">
      <span class="hdr">Drag on each UMAP</span>
      <label class="toggle-mesh"><input type="radio" name="plot-drag-mode" value="zoom" checked /> Zoom into box (only the plot you drag)</label>
      <label class="toggle-mesh"><input type="radio" name="plot-drag-mode" value="select" /> Box select — highlight pairs (amber, same as pinned)</label>
      <label class="toggle-mesh"><input type="radio" name="plot-drag-mode" value="paint" /> Paint region — draw a freeform outline (MS Paint–style)</label>
    </div>
    <div class="toolbar toolbar-row pair-table-controls">
      <span class="hdr">Pair tables</span>
      <label for="pair-table-sort">Sort by</label>
      <select id="pair-table-sort" aria-label="Sort mapped pair tables">
        <option value="corr_desc" selected>Correlation (high → low)</option>
        <option value="corr_asc">Correlation (low → high)</option>
        <option value="aid">A feature id</option>
        <option value="bid">B feature id</option>
      </select>
      <label class="toggle-mesh"><input type="checkbox" id="corr-filter-global-check" /> Global: corr ≥</label>
      <input type="number" id="global-corr-min" step="0.01" min="-1" max="1" value="" placeholder="0.2" title="When checked, hide pairs below this B→A correlation (selection table + base for category view)" style="width:76px;padding:5px 8px;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#e6edf3;" />
      <label class="toggle-mesh"><input type="checkbox" id="corr-filter-category-check" /> Category: corr ≥</label>
      <input type="number" id="category-corr-min" step="0.01" min="-1" max="1" value="" placeholder="0.3" title="When a semantic group is selected and this is checked, category table / purple mesh / violet highlights use this extra floor (AND with global if global is checked)" style="width:76px;padding:5px 8px;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#e6edf3;" />
    </div>
    <div class="row">
      <div class="panel">
        <p class="title">Model A</p>
        <div class="plot-panel" id="plot-panel-a">
          <div id="plot-a" class="plot"></div>
          <canvas id="paint-canvas-a" class="paint-select-canvas" width="8" height="8" aria-hidden="true"></canvas>
        </div>
        <div id="detail-a" class="details">Hover a feature point.</div>
      </div>
      <div class="panel">
        <p class="title">Model B</p>
        <div class="plot-panel" id="plot-panel-b">
          <div id="plot-b" class="plot"></div>
          <canvas id="paint-canvas-b" class="paint-select-canvas" width="8" height="8" aria-hidden="true"></canvas>
        </div>
        <div id="detail-b" class="details">Hover a feature point.</div>
      </div>
    </div>
    <div id="category-table-wrap" class="cat-table-wrap" style="display:none;">
      <p class="tbl-title" id="category-table-title"></p>
      <div class="cat-table-scroll">
        <table class="semantic-pairs" aria-label="Mapped feature pairs for selected category">
          <thead>
            <tr>
              <th>A feature</th>
              <th>A label</th>
              <th>B feature</th>
              <th>B label</th>
              <th>Corr (B→A)</th>
            </tr>
          </thead>
          <tbody id="category-pair-tbody"></tbody>
        </table>
      </div>
    </div>
    <div id="selection-table-wrap" class="cat-table-wrap" style="display:none;">
      <p class="tbl-title" id="selection-table-title"></p>
      <div class="cat-table-scroll">
        <table class="semantic-pairs" aria-label="Selected feature pairs (pinned and region selection)">
          <thead>
            <tr>
              <th>A feature</th>
              <th>A label</th>
              <th>B feature</th>
              <th>B label</th>
              <th>Corr (B→A)</th>
            </tr>
          </thead>
          <tbody id="selection-pair-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    const payload = {payload_js};
    const aPts = payload.model_a.points;
    const bPts = payload.model_b.points;
    const semanticGroups = payload.semantic_groups || [];
    const idxByAId = Object.fromEntries(aPts.map((p, i) => [String(p.feature_id), i]));
    const idxByBId = Object.fromEntries(bPts.map((p, i) => [String(p.feature_id), i]));

    const catSelect = document.getElementById("category-select");
    const categoryTableWrap = document.getElementById("category-table-wrap");
    const categoryTableTitle = document.getElementById("category-table-title");
    const categoryPairTbody = document.getElementById("category-pair-tbody");
    const selectionTableWrap = document.getElementById("selection-table-wrap");
    const selectionTableTitle = document.getElementById("selection-table-title");
    const selectionPairTbody = document.getElementById("selection-pair-tbody");
    const categoryMetricsHost = document.getElementById("category-metrics");
    catSelect.appendChild(new Option("(none)", ""));
    for (const g of semanticGroups) {{
      const na = (g.feature_ids_a || []).length;
      const nb = (g.feature_ids_b || []).length;
      if (na + nb === 0) continue;
      catSelect.appendChild(new Option(`${{g.name}} (A:${{na}}, B:${{nb}})`, g.id));
    }}

    function escHtml(s) {{
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }}

    (function renderMetricsSummary() {{
      const host = document.getElementById("metrics-summary");
      if (!host) return;
      const m = payload.meta && payload.meta.metrics;
      if (!m) {{
        host.innerHTML = "<p style='margin:0;color:#8b949e;'>No <code>meta.metrics</code> in this JSON (re-run <code>pythia_feature_mapping_viz.py</code> to populate similarity metrics).</p>";
        return;
      }}
      function fmt(x) {{
        if (x == null || typeof x !== "number" || !Number.isFinite(x)) return "—";
        return x.toFixed(4);
      }}
      const rows = [
        ["Alignment method", String(m.mapping_method || "—")],
        ["Mean |corr| at map (full activation pool)", fmt(m.mean_abs_pool_corr_at_map)],
        ["Mean corr at map (full activation pool)", fmt(m.mean_pool_corr_at_map)],
        ["Linear CKA (activations, selected features)", fmt(m.linear_cka_activations)],
        ["RSA Spearman (activation RDMs, selected)", fmt(m.rsa_spearman_activation_rdm)],
        ["Orthogonal Procrustes rel. RMSE (activations, selected)", fmt(m.procrustes_rel_rmse_activations)],
        ["Mean Pearson (per matched column, token axis)", fmt(m.mean_matched_column_pearson)],
        ["SVCCA (paired decoder directions)", fmt(m.svcca_decoder_paired)],
        ["SVCCA (paired activation trajectories)", fmt(m.svcca_activation_paired)],
      ];
      if (m.ot_primal_cost_sinkhorn != null && typeof m.ot_primal_cost_sinkhorn === "number" && Number.isFinite(m.ot_primal_cost_sinkhorn)) {{
        rows.push(["OT primal cost ⟨T, 1−C⟩ (Sinkhorn)", fmt(m.ot_primal_cost_sinkhorn)]);
        rows.push(["OT entropic regularization", fmt(m.ot_reg)]);
      }}
      const tbl = '<table>' + rows.map(([k, v]) =>
        '<tr><td class="k">' + escHtml(k) + '</td><td class="v">' + escHtml(String(v)) + '</td></tr>'
      ).join('') + '</table>';
      const note = "<p style='margin:8px 0 0 0;color:#8b949e;font-size:11px;line-height:1.45;'>"
        + "CKA and RSA use the same token positions for selected model-B features and their mapped model-A columns. "
        + "Procrustes applies an orthogonal map in sample space to best align those matched activation profiles (lower relative RMSE means closer geometry). "
        + "SVCCA on decoders runs when the number of paired features exceeds both residual widths (same convention as Lan et al.); otherwise activation SVCCA may still apply when there are many more tokens than features. "
        + "Hungarian alignment is a one-to-one maximum-total-correlation assignment over the pooled features (equal pool sizes).</p>";
      host.innerHTML = "<h4>Quantitative feature-space similarity</h4>" + tbl + note;
    }})();

    function fmtMetricCell(x) {{
      if (x == null || typeof x !== "number" || !Number.isFinite(x)) return "—";
      return x.toFixed(4);
    }}

    function renderCategoryMetrics(groupId) {{
      if (!categoryMetricsHost) return;
      if (!groupId) {{
        categoryMetricsHost.style.display = "none";
        categoryMetricsHost.innerHTML = "";
        return;
      }}
      const g = semanticGroups.find((x) => x.id === groupId);
      if (!g) {{
        categoryMetricsHost.style.display = "none";
        categoryMetricsHost.innerHTML = "";
        return;
      }}
      const cm = g.metrics;
      if (!cm) {{
        categoryMetricsHost.style.display = "block";
        categoryMetricsHost.innerHTML = "<h4>Category metrics</h4><p style='margin:0;color:#8b949e;font-size:12px;'>No per-category metrics in this JSON (re-run <code>pythia_feature_mapping_viz.py</code>).</p>";
        return;
      }}
      const n = cm.n_pairs != null ? String(cm.n_pairs) : "—";
      const subRows = [
        ["Paired features (mapped A↔B in this category)", n],
        ["Linear CKA (activations)", fmtMetricCell(cm.linear_cka_activations)],
        ["RSA Spearman (activation RDMs)", fmtMetricCell(cm.rsa_spearman_activation_rdm)],
        ["Orthogonal Procrustes rel. RMSE (activations)", fmtMetricCell(cm.procrustes_rel_rmse_activations)],
        ["Mean Pearson (per matched column)", fmtMetricCell(cm.mean_matched_column_pearson)],
        ["SVCCA (paired decoder directions)", fmtMetricCell(cm.svcca_decoder_paired)],
        ["SVCCA (paired activation trajectories)", fmtMetricCell(cm.svcca_activation_paired)],
      ];
      const tbl = '<table>' + subRows.map(([k, v]) =>
        '<tr><td class="k">' + escHtml(k) + '</td><td class="v">' + escHtml(String(v)) + '</td></tr>'
      ).join('') + '</table>';
      categoryMetricsHost.innerHTML = "<h4>Metrics: " + escHtml(g.name) + "</h4>" + tbl
        + "<p style='margin:8px 0 0 0;color:#8b949e;font-size:11px;line-height:1.45;'>"
        + "Subset includes model-B features in this semantic bucket (plus model-A features in the bucket when they link to a selected mapped pair). "
        + "SVCCA is omitted for category subsets in the pipeline build (too slow for many groups); CKA / RSA / Procrustes / Pearson still reflect this subset.</p>";
      categoryMetricsHost.style.display = "block";
    }}

    function readNumericOrNull(el) {{
      if (!el) return null;
      const v = parseFloat(el.value);
      return Number.isFinite(v) ? v : null;
    }}

    function readGlobalCorrThreshold() {{
      const chk = document.getElementById("corr-filter-global-check");
      if (!chk || !chk.checked) return null;
      return readNumericOrNull(document.getElementById("global-corr-min"));
    }}

    function readCategoryCorrThreshold() {{
      const chk = document.getElementById("corr-filter-category-check");
      const gid = catSelect.value || "";
      if (!chk || !chk.checked || !gid) return null;
      return readNumericOrNull(document.getElementById("category-corr-min"));
    }}

    function pairPassesCorr(corr, globalT, categoryT) {{
      if (!Number.isFinite(corr)) return false;
      if (globalT != null && corr < globalT) return false;
      if (categoryT != null && corr < categoryT) return false;
      return true;
    }}

    function sortPairRows(rows) {{
      const sel = document.getElementById("pair-table-sort");
      const mode = sel && sel.value ? sel.value : "corr_desc";
      const out = rows.slice();
      if (mode === "corr_desc") {{
        out.sort((a, b) => (Number(b.corr) - Number(a.corr)) || (a.aid - b.aid) || (a.bid - b.bid));
      }} else if (mode === "corr_asc") {{
        out.sort((a, b) => (Number(a.corr) - Number(b.corr)) || (a.aid - b.aid) || (a.bid - b.bid));
      }} else if (mode === "bid") {{
        out.sort((a, b) => (a.bid - b.bid) || (a.aid - b.aid));
      }} else {{
        out.sort((a, b) => (a.aid - b.aid) || (a.bid - b.bid));
      }}
      return out;
    }}

    function collectCategoryPairRows(g) {{
      const setA = new Set(g.feature_ids_a || []);
      const setB = new Set(g.feature_ids_b || []);
      const rows = [];
      const seen = new Set();
      function addRow(aid, al, bid, bl, corr) {{
        const k = `${{aid}}:${{bid}}`;
        if (seen.has(k)) return;
        seen.add(k);
        rows.push({{ aid, al, bid, bl, corr }});
      }}
      for (const p of aPts) {{
        if (!setA.has(p.feature_id)) continue;
        const mid = p.mapped_feature_id;
        if (mid == null || idxByBId[String(mid)] === undefined) continue;
        const q = bPts[idxByBId[String(mid)]];
        addRow(p.feature_id, p.label, mid, q.label, q.corr_to_mapped);
      }}
      for (const q of bPts) {{
        if (!setB.has(q.feature_id)) continue;
        const mid = q.mapped_feature_id;
        if (mid == null || idxByAId[String(mid)] === undefined) continue;
        const p = aPts[idxByAId[String(mid)]];
        addRow(mid, p.label, q.feature_id, q.label, q.corr_to_mapped);
      }}
      return rows;
    }}

    function refreshPairTablesFiltersAndMesh() {{
      rebuildCategoryTable(catSelect.value || "");
      rebuildUnifiedPairsTable();
      if (catSelect.value) paintCategoryHighlights();
      else updateLineSubsetShapes();
    }}

    function rebuildCategoryTable(groupId) {{
      if (!groupId) {{
        categoryTableWrap.style.display = "none";
        categoryPairTbody.innerHTML = "";
        categoryTableTitle.textContent = "";
        updateLineSubsetShapes();
        return;
      }}
      const g = semanticGroups.find((x) => x.id === groupId);
      if (!g) {{
        categoryTableWrap.style.display = "none";
        updateLineSubsetShapes();
        return;
      }}
      let rows = collectCategoryPairRows(g);
      const nTotal = rows.length;
      const gThr = readGlobalCorrThreshold();
      const cThr = readCategoryCorrThreshold();
      rows = rows.filter((r) => pairPassesCorr(r.corr, gThr, cThr));
      rows = sortPairRows(rows);
      const filt = gThr != null || cThr != null;
      categoryTableTitle.textContent = filt
        ? `Mapped pairs involving "${{g.name}}" (${{rows.length}} of ${{nTotal}} after corr filters)`
        : `Mapped pairs involving "${{g.name}}" (${{rows.length}} pairs)`;
      categoryPairTbody.innerHTML = rows
        .map(
          (r) =>
            `<tr class="pair-row" data-aid="${{r.aid}}" data-bid="${{r.bid}}"><td class="num">${{r.aid}}</td><td>${{escHtml(r.al)}}</td><td class="num">${{r.bid}}</td><td>${{escHtml(r.bl)}}</td><td class="corr">${{Number.isFinite(r.corr) ? r.corr.toFixed(4) : "—"}}</td></tr>`
        )
        .join("");
      categoryTableWrap.style.display = "block";
      syncPinnedTableRows();
      updateLineSubsetShapes();
    }}

    const baseA = {{
      type: "scatter",
      mode: "markers",
      x: aPts.map(p => p.x),
      y: aPts.map(p => p.y),
      marker: {{ size: 5, color: "#58a6ff", opacity: 0.7 }},
      selected: {{ marker: {{ size: 5, color: "#58a6ff", opacity: 0.7, line: {{ width: 0, color: "#58a6ff" }} }} }},
      unselected: {{ marker: {{ size: 5, color: "#58a6ff", opacity: 0.7 }} }},
      customdata: aPts.map(p => [p.feature_id, p.label, p.mapped_feature_id]),
      hovertemplate: "Feature %{{customdata[0]}}<br>%{{customdata[1]}}<extra></extra>",
      name: "Model A features",
    }};
    const catHiA = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 10, color: "#a371f7", line: {{ width: 1.5, color: "#e6edf3" }}, opacity: 0.95 }},
      hoverinfo: "skip",
      showlegend: false,
      name: "category-a",
    }};
    const hoverHiA = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 12, color: "#f78166", line: {{ width: 1, color: "#ffffff" }} }},
      hoverinfo: "skip",
      showlegend: false,
      name: "hover-a",
    }};
    const pinHiA = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 14, color: "#ffc658", line: {{ width: 2, color: "#0d1117" }} }},
      hoverinfo: "skip",
      showlegend: false,
      name: "pinned-a",
    }};

    const marqueeHiA = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 14, color: "#ffc658", line: {{ width: 2, color: "#0d1117" }} }},
      hoverinfo: "skip",
      showlegend: false,
      name: "selection-a",
    }};

    const baseB = {{
      type: "scatter",
      mode: "markers",
      x: bPts.map(p => p.x),
      y: bPts.map(p => p.y),
      marker: {{ size: 5, color: "#a5d6ff", opacity: 0.7 }},
      selected: {{ marker: {{ size: 5, color: "#a5d6ff", opacity: 0.7, line: {{ width: 0, color: "#a5d6ff" }} }} }},
      unselected: {{ marker: {{ size: 5, color: "#a5d6ff", opacity: 0.7 }} }},
      customdata: bPts.map(p => [p.feature_id, p.label, p.mapped_feature_id, p.corr_to_mapped]),
      hovertemplate: "Feature %{{customdata[0]}}<br>%{{customdata[1]}}<br>corr=%{{customdata[3]:.3f}}<extra></extra>",
      name: "Model B features",
    }};
    const catHiB = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 10, color: "#a371f7", line: {{ width: 1.5, color: "#e6edf3" }}, opacity: 0.95 }},
      hoverinfo: "skip",
      showlegend: false,
      name: "category-b",
    }};
    const hoverHiB = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 12, color: "#f78166", line: {{ width: 1, color: "#ffffff" }} }},
      hoverinfo: "skip",
      showlegend: false,
      name: "hover-b",
    }};
    const pinHiB = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 14, color: "#ffc658", line: {{ width: 2, color: "#0d1117" }} }},
      hoverinfo: "skip",
      showlegend: false,
      name: "pinned-b",
    }};

    const marqueeHiB = {{
      type: "scatter",
      mode: "markers",
      x: [],
      y: [],
      marker: {{ size: 14, color: "#ffc658", line: {{ width: 2, color: "#0d1117" }} }},
      hoverinfo: "skip",
      showlegend: false,
      name: "selection-b",
    }};

    const layoutBase = {{
      paper_bgcolor: "#161b22",
      plot_bgcolor: "#0d1117",
      margin: {{ t: 20, l: 35, r: 15, b: 35 }},
      xaxis: {{ title: "UMAP-1", gridcolor: "#30363d", zeroline: false }},
      yaxis: {{ title: "UMAP-2", gridcolor: "#30363d", zeroline: false }},
      font: {{ color: "#e6edf3", size: 11 }},
      showlegend: false
    }};
    const layoutA = {{
      ...layoutBase,
      margin: {{ ...layoutBase.margin }},
      font: {{ ...layoutBase.font }},
      xaxis: {{ ...layoutBase.xaxis }},
      yaxis: {{ ...layoutBase.yaxis }},
      title: payload.model_a.name,
      uirevision: "umap-model-a",
      dragmode: "zoom",
    }};
    const layoutB = {{
      ...layoutBase,
      margin: {{ ...layoutBase.margin }},
      font: {{ ...layoutBase.font }},
      xaxis: {{ ...layoutBase.xaxis }},
      yaxis: {{ ...layoutBase.yaxis }},
      title: payload.model_b.name,
      uirevision: "umap-model-b",
      dragmode: "zoom",
    }};

    const config = {{ responsive: true, displayModeBar: true }};
    Plotly.newPlot("plot-a", [baseA, catHiA, hoverHiA, pinHiA, marqueeHiA], {{ ...layoutA, shapes: [] }}, config);
    Plotly.newPlot("plot-b", [baseB, catHiB, hoverHiB, pinHiB, marqueeHiB], {{ ...layoutB, shapes: [] }}, config);

    const plotA = document.getElementById("plot-a");
    const plotB = document.getElementById("plot-b");
    const plotPanelA = document.getElementById("plot-panel-a");
    const plotPanelB = document.getElementById("plot-panel-b");
    const paintCanvasA = document.getElementById("paint-canvas-a");
    const paintCanvasB = document.getElementById("paint-canvas-b");
    const detailA = document.getElementById("detail-a");
    const detailB = document.getElementById("detail-b");
    const resetPinsBtn = document.getElementById("reset-pins-btn");
    const toggleMeshPinned = document.getElementById("toggle-mesh-pinned");
    const toggleMeshCategory = document.getElementById("toggle-mesh-category");
    const meshHint = document.getElementById("mesh-hint");

    let categoryIdLocked = "";
    const CATEGORY_TRACE = 1;
    const HOVER_TRACE = 2;
    const PINNED_TRACE = 3;
    const SELECTION_OVERLAY_TRACE = 4;
    const pinnedPairKeys = new Set();
    const selectionPairKeys = new Set();

    function pairKey(aid, bid) {{
      return `${{aid}}:${{bid}}`;
    }}

    function paintPinnedHighlights() {{
      const xa = [];
      const ya = [];
      const xb = [];
      const yb = [];
      for (const k of pinnedPairKeys) {{
        const parts = k.split(":");
        const aid = Number(parts[0]);
        const bid = Number(parts[1]);
        const ia = idxByAId[String(aid)];
        const ib = idxByBId[String(bid)];
        if (ia !== undefined) {{
          xa.push(aPts[ia].x);
          ya.push(aPts[ia].y);
        }}
        if (ib !== undefined) {{
          xb.push(bPts[ib].x);
          yb.push(bPts[ib].y);
        }}
      }}
      Plotly.restyle(plotA, {{ x: [xa], y: [ya] }}, [PINNED_TRACE]);
      Plotly.restyle(plotB, {{ x: [xb], y: [yb] }}, [PINNED_TRACE]);
    }}

    function syncPinnedTableRows() {{
      document.querySelectorAll("#category-pair-tbody tr.pair-row, #selection-pair-tbody tr.pair-row").forEach((tr) => {{
        const aid = Number(tr.dataset.aid);
        const bid = Number(tr.dataset.bid);
        const sel = !Number.isNaN(aid) && !Number.isNaN(bid) && pinnedPairKeys.has(pairKey(aid, bid));
        tr.classList.toggle("pair-row-selected", sel);
      }});
    }}

    function getSelectionAIdsFromPairs() {{
      const s = new Set();
      for (const k of selectionPairKeys) s.add(Number(k.split(":")[0]));
      return [...s];
    }}

    function getSelectionBIdsFromPairs() {{
      const s = new Set();
      for (const k of selectionPairKeys) s.add(Number(k.split(":")[1]));
      return [...s];
    }}

    function mergeSortedUniqueIds(a, b) {{
      return [...new Set([...(a || []), ...(b || [])])].sort((x, y) => x - y);
    }}

    function rebuildUnifiedPairsTable() {{
      const keys = new Set([...pinnedPairKeys, ...selectionPairKeys]);
      if (keys.size === 0) {{
        selectionTableWrap.style.display = "none";
        selectionPairTbody.innerHTML = "";
        selectionTableTitle.textContent = "";
        syncPinnedTableRows();
        return;
      }}
      const rows = [];
      const gThrOnly = readGlobalCorrThreshold();
      for (const k of keys) {{
        const parts = k.split(":");
        const aid = Number(parts[0]);
        const bid = Number(parts[1]);
        const ia = idxByAId[String(aid)];
        const ib = idxByBId[String(bid)];
        if (ia === undefined || ib === undefined) continue;
        const p = aPts[ia];
        const q = bPts[ib];
        const corr = q.corr_to_mapped;
        if (!pairPassesCorr(corr, gThrOnly, null)) continue;
        rows.push({{ aid, al: p.label, bid, bl: q.label, corr }});
      }}
      const sorted = sortPairRows(rows);
      const nAll = [...keys].length;
      const filt = gThrOnly != null;
      selectionTableTitle.textContent = filt
        ? `Selected feature pairs (${{sorted.length}} of ${{nAll}} after global corr filter) — pins + box/paint; click row to pin/unpin`
        : `Selected feature pairs (${{sorted.length}}) — from click pins and from box/paint; click a row to pin/unpin`;
      selectionPairTbody.innerHTML = sorted
        .map(
          (r) =>
            `<tr class="pair-row" data-aid="${{r.aid}}" data-bid="${{r.bid}}"><td class="num">${{r.aid}}</td><td>${{escHtml(r.al)}}</td><td class="num">${{r.bid}}</td><td>${{escHtml(r.bl)}}</td><td class="corr">${{Number.isFinite(r.corr) ? r.corr.toFixed(4) : "—"}}</td></tr>`
        )
        .join("");
      selectionTableWrap.style.display = "block";
      syncPinnedTableRows();
    }}

    function togglePinForPair(aid, bid) {{
      if (bid == null || aid == null) return;
      const k = pairKey(aid, bid);
      if (pinnedPairKeys.has(k)) pinnedPairKeys.delete(k);
      else pinnedPairKeys.add(k);
      paintPinnedHighlights();
      updateLineSubsetShapes();
      rebuildUnifiedPairsTable();
    }}

    function resetAllPins() {{
      paintStroke = null;
      pinnedPairKeys.clear();
      Plotly.restyle(plotA, {{ x: [[]], y: [[]] }}, [PINNED_TRACE]);
      Plotly.restyle(plotB, {{ x: [[]], y: [[]] }}, [PINNED_TRACE]);
      fullClearSelectionState();
      clearPaintCanvas(paintCanvasA);
      clearPaintCanvas(paintCanvasB);
      clearPlotlySelectionChromeBoth();
      syncPinnedTableRows();
      updateLineSubsetShapes();
    }}

    if (resetPinsBtn) resetPinsBtn.addEventListener("click", () => resetAllPins());

    categoryTableWrap.addEventListener("click", (ev) => {{
      const tr = ev.target.closest("#category-pair-tbody tr.pair-row");
      if (!tr) return;
      const aid = Number(tr.dataset.aid);
      const bid = Number(tr.dataset.bid);
      if (Number.isNaN(aid) || Number.isNaN(bid)) return;
      togglePinForPair(aid, bid);
    }});

    selectionTableWrap.addEventListener("click", (ev) => {{
      const tr = ev.target.closest("#selection-pair-tbody tr.pair-row");
      if (!tr) return;
      const aid = Number(tr.dataset.aid);
      const bid = Number(tr.dataset.bid);
      if (Number.isNaN(aid) || Number.isNaN(bid)) return;
      togglePinForPair(aid, bid);
    }});

    function idsToCoords(ids, pts, idxById) {{
      const xs = [];
      const ys = [];
      for (const id of ids) {{
        const ix = idxById[String(id)];
        if (ix !== undefined) {{
          xs.push(pts[ix].x);
          ys.push(pts[ix].y);
        }}
      }}
      return [xs, ys];
    }}

    const MAX_MESH_POINTS = 32;

    function coordsFromIds(ids, pts, idxById) {{
      const out = [];
      for (const id of ids) {{
        const ix = idxById[String(id)];
        if (ix !== undefined) out.push({{ x: pts[ix].x, y: pts[ix].y }});
      }}
      return out;
    }}

    function capSortedUniqueIds(ids, maxN) {{
      const u = [...new Set(ids)].sort((a, b) => a - b);
      if (u.length <= maxN) return {{ ids: u, capped: false }};
      return {{ ids: u.slice(0, maxN), capped: true }};
    }}

    function meshLineShapes(coords, lineOptions) {{
      if (coords.length < 2) return [];
      const shapes = [];
      for (let i = 0; i < coords.length; i++) {{
        for (let j = i + 1; j < coords.length; j++) {{
          const p = coords[i];
          const q = coords[j];
          shapes.push({{
            type: "line",
            xref: "x",
            yref: "y",
            x0: p.x,
            x1: q.x,
            y0: p.y,
            y1: q.y,
            line: lineOptions,
            layer: "below",
          }});
        }}
      }}
      return shapes;
    }}

    function getPinnedAIds() {{
      const s = new Set();
      for (const k of pinnedPairKeys) s.add(Number(k.split(":")[0]));
      return [...s];
    }}

    function getPinnedBIds() {{
      const s = new Set();
      for (const k of pinnedPairKeys) s.add(Number(k.split(":")[1]));
      return [...s];
    }}

    function updateLineSubsetShapes() {{
      const wantPin = toggleMeshPinned.checked;
      const wantCat = toggleMeshCategory.checked && !!categoryIdLocked;
      let cappedNote = false;
      const shapesA = [];
      const shapesB = [];
      const linePin = {{ color: "rgba(57,211,83,0.55)", width: 1.5 }};
      const lineCat = {{ color: "rgba(136,119,232,0.45)", width: 1.25 }};
      if (wantPin) {{
        const mergedA = mergeSortedUniqueIds(getPinnedAIds(), getSelectionAIdsFromPairs());
        const mergedB = mergeSortedUniqueIds(getPinnedBIds(), getSelectionBIdsFromPairs());
        const pa = capSortedUniqueIds(mergedA, MAX_MESH_POINTS);
        const pb = capSortedUniqueIds(mergedB, MAX_MESH_POINTS);
        shapesA.push(...meshLineShapes(coordsFromIds(pa.ids, aPts, idxByAId), linePin));
        shapesB.push(...meshLineShapes(coordsFromIds(pb.ids, bPts, idxByBId), linePin));
        if (pa.capped || pb.capped) cappedNote = true;
      }}
      if (wantCat) {{
        const g = semanticGroups.find((x) => x.id === categoryIdLocked);
        if (g) {{
          let rows = collectCategoryPairRows(g);
          const gThr = readGlobalCorrThreshold();
          const cThr = readCategoryCorrThreshold();
          rows = rows.filter((r) => pairPassesCorr(r.corr, gThr, cThr));
          const fa = [...new Set(rows.map((r) => r.aid))].sort((x, y) => x - y);
          const fb = [...new Set(rows.map((r) => r.bid))].sort((x, y) => x - y);
          const ca = capSortedUniqueIds(fa, MAX_MESH_POINTS);
          const cb = capSortedUniqueIds(fb, MAX_MESH_POINTS);
          shapesA.push(...meshLineShapes(coordsFromIds(ca.ids, aPts, idxByAId), lineCat));
          shapesB.push(...meshLineShapes(coordsFromIds(cb.ids, bPts, idxByBId), lineCat));
          if (ca.capped || cb.capped) cappedNote = true;
        }}
      }}
      meshHint.textContent = cappedNote
        ? `Subset mesh: at most ${{MAX_MESH_POINTS}} points per side per source (sorted id) for responsiveness.`
        : wantPin || wantCat
          ? "Subset mesh: green = selected features (click‑pinned + box/paint region); purple = category (complete graph on each UMAP)."
          : "";
      Plotly.relayout(plotA, {{ shapes: shapesA }});
      Plotly.relayout(plotB, {{ shapes: shapesB }});
    }}

    [toggleMeshPinned, toggleMeshCategory].forEach((el) => {{
      el.addEventListener("change", () => updateLineSubsetShapes());
    }});

    [["pair-table-sort", "change"], ["global-corr-min", "input"], ["category-corr-min", "input"], ["corr-filter-global-check", "change"], ["corr-filter-category-check", "change"]].forEach(([id, ev]) => {{
      const el = document.getElementById(id);
      if (el) el.addEventListener(ev, () => refreshPairTablesFiltersAndMesh());
    }});

    function clearHoverHighlightsOnly() {{
      Plotly.restyle(plotA, {{ x: [[]], y: [[]] }}, [HOVER_TRACE]);
      Plotly.restyle(plotB, {{ x: [[]], y: [[]] }}, [HOVER_TRACE]);
    }}

    function paintCategoryHighlights() {{
      const g = semanticGroups.find((x) => x.id === categoryIdLocked);
      if (!g) return;
      let rows = collectCategoryPairRows(g);
      const gThr = readGlobalCorrThreshold();
      const cThr = readCategoryCorrThreshold();
      rows = rows.filter((r) => pairPassesCorr(r.corr, gThr, cThr));
      const idA = [...new Set(rows.map((r) => r.aid))];
      const idB = [...new Set(rows.map((r) => r.bid))];
      const [xa, ya] = idsToCoords(idA, aPts, idxByAId);
      const [xb, yb] = idsToCoords(idB, bPts, idxByBId);
      Plotly.restyle(plotA, {{ x: [xa], y: [ya] }}, [CATEGORY_TRACE]);
      Plotly.restyle(plotB, {{ x: [xb], y: [yb] }}, [CATEGORY_TRACE]);
      detailA.textContent = `${{g.name}}: ${{xa.length}} on A, ${{xb.length}} on B (violet, after corr filters). Orange = hover. Amber = click‑pinned pairs.`;
      detailB.textContent = "Table: click a row to pin/unpin that pair on the maps.";
      updateLineSubsetShapes();
    }}

    function clearCategoryAndHover() {{
      Plotly.restyle(plotA, {{ x: [[], []], y: [[], []] }}, [CATEGORY_TRACE, HOVER_TRACE]);
      Plotly.restyle(plotB, {{ x: [[], []], y: [[], []] }}, [CATEGORY_TRACE, HOVER_TRACE]);
    }}

    function onPlotUnhover() {{
      clearHoverHighlightsOnly();
      if (categoryIdLocked) {{
        const g = semanticGroups.find((x) => x.id === categoryIdLocked);
        if (g) {{
          detailA.textContent = `${{g.name}}: violet = category (stays on). Orange = hover. Amber = click‑pinned.`;
          detailB.textContent = "Click a map point or table row to pin/unpin a pair.";
        }}
      }} else {{
        detailA.textContent = "Hover a feature point. Click a mapped pair on a plot to pin (amber); click again to unpin.";
        detailB.textContent = "Use Reset all selected features to clear amber pins.";
      }}
    }}

    catSelect.addEventListener("change", () => {{
      categoryIdLocked = catSelect.value || "";
      clearHoverHighlightsOnly();
      if (!categoryIdLocked) {{
        clearCategoryAndHover();
        detailA.textContent = "Hover a feature point. Click a mapped pair on a plot to pin (amber); click again to unpin.";
        detailB.textContent = "Use Reset all selected features to clear amber pins.";
        renderCategoryMetrics("");
      }} else {{
        renderCategoryMetrics(categoryIdLocked);
      }}
      refreshPairTablesFiltersAndMesh();
    }});

    function renderDetails(aPoint, bPoint) {{
      detailA.textContent = aPoint
        ? `Feature ${{aPoint.feature_id}}\\nLabel: ${{aPoint.label || "<none>"}}\\nMapped to B: ${{aPoint.mapped_feature_id ?? "none"}}`
        : "Hover a feature point.";
      detailB.textContent = bPoint
        ? `Feature ${{bPoint.feature_id}}\\nLabel: ${{bPoint.label || "<none>"}}\\nMapped to A: ${{bPoint.mapped_feature_id ?? "none"}}\\nCorrelation: ${{(bPoint.corr_to_mapped ?? NaN).toFixed(4)}}`
        : "Hover a feature point.";
    }}

    function highlightPairFromA(aIdx) {{
      const aPoint = aPts[aIdx];
      const bIdx = aPoint.mapped_feature_id == null ? undefined : idxByBId[String(aPoint.mapped_feature_id)];
      const bPoint = bIdx === undefined ? null : bPts[bIdx];
      Plotly.restyle(plotA, {{ x: [[aPoint.x]], y: [[aPoint.y]] }}, [HOVER_TRACE]);
      if (bPoint) {{
        Plotly.restyle(plotB, {{ x: [[bPoint.x]], y: [[bPoint.y]] }}, [HOVER_TRACE]);
      }} else {{
        Plotly.restyle(plotB, {{ x: [[]], y: [[]] }}, [HOVER_TRACE]);
      }}
      renderDetails(aPoint, bPoint);
    }}

    function highlightPairFromB(bIdx) {{
      const bPoint = bPts[bIdx];
      const aIdx = bPoint.mapped_feature_id == null ? undefined : idxByAId[String(bPoint.mapped_feature_id)];
      const aPoint = aIdx === undefined ? null : aPts[aIdx];
      Plotly.restyle(plotB, {{ x: [[bPoint.x]], y: [[bPoint.y]] }}, [HOVER_TRACE]);
      if (aPoint) {{
        Plotly.restyle(plotA, {{ x: [[aPoint.x]], y: [[aPoint.y]] }}, [HOVER_TRACE]);
      }} else {{
        Plotly.restyle(plotA, {{ x: [[]], y: [[]] }}, [HOVER_TRACE]);
      }}
      renderDetails(aPoint, bPoint);
    }}

    plotA.on("plotly_click", (ev) => {{
      const point = ev.points?.[0];
      if (!point || point.curveNumber !== 0) return;
      const p = aPts[point.pointIndex];
      if (p.mapped_feature_id == null) return;
      togglePinForPair(p.feature_id, p.mapped_feature_id);
    }});
    plotB.on("plotly_click", (ev) => {{
      const point = ev.points?.[0];
      if (!point || point.curveNumber !== 0) return;
      const q = bPts[point.pointIndex];
      if (q.mapped_feature_id == null) return;
      togglePinForPair(q.mapped_feature_id, q.feature_id);
    }});

    plotA.on("plotly_hover", (ev) => {{
      const point = ev.points?.[0];
      if (!point || point.curveNumber !== 0) return;
      highlightPairFromA(point.pointIndex);
    }});
    plotB.on("plotly_hover", (ev) => {{
      const point = ev.points?.[0];
      if (!point || point.curveNumber !== 0) return;
      highlightPairFromB(point.pointIndex);
    }});
    plotA.on("plotly_unhover", onPlotUnhover);
    plotB.on("plotly_unhover", onPlotUnhover);

    function getPlotDragMode() {{
      const el = document.querySelector('input[name="plot-drag-mode"]:checked');
      return el ? el.value : "zoom";
    }}

    function resizePaintCanvases() {{
      [paintCanvasA, paintCanvasB].forEach((canvas) => {{
        const panel = canvas.parentElement;
        if (!panel) return;
        const rect = panel.getBoundingClientRect();
        const w = Math.max(1, Math.round(rect.width));
        const h = Math.max(1, Math.round(rect.height));
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${{w}}px`;
        canvas.style.height = `${{h}}px`;
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }});
    }}

    function syncPaintCanvasPointerEvents() {{
      const paint = getPlotDragMode() === "paint";
      paintCanvasA.style.pointerEvents = paint ? "auto" : "none";
      paintCanvasB.style.pointerEvents = paint ? "auto" : "none";
      if (paint) resizePaintCanvases();
    }}

    function clearPaintCanvas(c) {{
      const ctx = c.getContext("2d");
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.restore();
    }}

    function screenToDataXY(gd, clientX, clientY) {{
      const r = gd.getBoundingClientRect();
      const px = clientX - r.left;
      const py = clientY - r.top;
      const l = gd._fullLayout;
      if (!l) return null;
      const xa = l.xaxis;
      const ya = l.yaxis;
      if (!xa || !ya || xa._length == null || ya._length == null) return null;
      const xr = xa.range;
      const yr = ya.range;
      if (!xr || xr.length < 2 || !yr || yr.length < 2) return null;
      const xd = xr[0] + ((px - xa._offset) / xa._length) * (xr[1] - xr[0]);
      const yd = yr[0] + (1 - (py - ya._offset) / ya._length) * (yr[1] - yr[0]);
      return {{ x: xd, y: yd }};
    }}

    function pointInPolygon(x, y, poly) {{
      let inside = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {{
        const xi = poly[i].x;
        const yi = poly[i].y;
        const xj = poly[j].x;
        const yj = poly[j].y;
        const cross = (yi > y) !== (yj > y);
        if (!cross) continue;
        const t = (yj - yi) || 1e-12;
        const xinters = ((xj - xi) * (y - yi)) / t + xi;
        if (x < xinters) inside = !inside;
      }}
      return inside;
    }}

    function clearPlotlySelectionChromeBoth() {{
      Plotly.restyle(
        plotA,
        {{
          selectedpoints: [null],
          selected: [{{ marker: {{ size: 5, color: "#58a6ff", opacity: 0.7, line: {{ width: 0, color: "#58a6ff" }} }} }}],
          unselected: [{{ marker: {{ size: 5, color: "#58a6ff", opacity: 0.7 }} }}],
        }},
        [0]
      );
      Plotly.restyle(
        plotB,
        {{
          selectedpoints: [null],
          selected: [{{ marker: {{ size: 5, color: "#a5d6ff", opacity: 0.7, line: {{ width: 0, color: "#a5d6ff" }} }} }}],
          unselected: [{{ marker: {{ size: 5, color: "#a5d6ff", opacity: 0.7 }} }}],
        }},
        [0]
      );
      const clr = {{ selections: [] }};
      Plotly.relayout(plotA, clr);
      Plotly.relayout(plotB, clr);
    }}

    function fullClearSelectionState() {{
      selectionPairKeys.clear();
      Plotly.restyle(plotA, {{ x: [[]], y: [[]] }}, [SELECTION_OVERLAY_TRACE]);
      Plotly.restyle(plotB, {{ x: [[]], y: [[]] }}, [SELECTION_OVERLAY_TRACE]);
      clearPlotlySelectionChromeBoth();
      rebuildUnifiedPairsTable();
      updateLineSubsetShapes();
    }}

    function applyRegionSelectionByIndices(plotGd, fromPlotA, indexList) {{
      const uniq = [...new Set(indexList)];
      if (uniq.length === 0) {{
        clearPlotlySelectionChromeBoth();
        return;
      }}
      selectionPairKeys.clear();
      const xa = [];
      const ya = [];
      const xb = [];
      const yb = [];
      if (fromPlotA) {{
        const seenIdx = new Set();
        const seenBId = new Set();
        for (const i of uniq) {{
          if (i < 0 || i >= aPts.length) continue;
          if (seenIdx.has(i)) continue;
          seenIdx.add(i);
          const ap = aPts[i];
          xa.push(ap.x);
          ya.push(ap.y);
          const mid = ap.mapped_feature_id;
          if (mid != null && idxByBId[String(mid)] !== undefined) {{
            selectionPairKeys.add(pairKey(ap.feature_id, mid));
          }}
          if (mid != null && !seenBId.has(String(mid)) && idxByBId[String(mid)] !== undefined) {{
            seenBId.add(String(mid));
            const bp = bPts[idxByBId[String(mid)]];
            xb.push(bp.x);
            yb.push(bp.y);
          }}
        }}
      }} else {{
        const seenIdx = new Set();
        const seenAId = new Set();
        for (const j of uniq) {{
          if (j < 0 || j >= bPts.length) continue;
          if (seenIdx.has(j)) continue;
          seenIdx.add(j);
          const bq = bPts[j];
          xb.push(bq.x);
          yb.push(bq.y);
          const ma = bq.mapped_feature_id;
          if (ma != null && idxByAId[String(ma)] !== undefined) {{
            selectionPairKeys.add(pairKey(ma, bq.feature_id));
          }}
          if (ma != null && !seenAId.has(String(ma)) && idxByAId[String(ma)] !== undefined) {{
            seenAId.add(String(ma));
            const ap = aPts[idxByAId[String(ma)]];
            xa.push(ap.x);
            ya.push(ap.y);
          }}
        }}
      }}
      Plotly.restyle(plotA, {{ x: [xa], y: [ya] }}, [SELECTION_OVERLAY_TRACE]);
      Plotly.restyle(plotB, {{ x: [xb], y: [yb] }}, [SELECTION_OVERLAY_TRACE]);
      clearPlotlySelectionChromeBoth();
      requestAnimationFrame(() => requestAnimationFrame(() => clearPlotlySelectionChromeBoth()));
      rebuildUnifiedPairsTable();
      updateLineSubsetShapes();
    }}

    function applyPlotDragMode() {{
      const m = getPlotDragMode();
      if (m === "zoom") {{
        fullClearSelectionState();
        clearPaintCanvas(paintCanvasA);
        clearPaintCanvas(paintCanvasB);
      }}
      let dragmode = "zoom";
      if (m === "select") dragmode = "select";
      else if (m === "paint") dragmode = "zoom";
      const relA = {{ dragmode }};
      const relB = {{ dragmode }};
      if (m === "select") {{
        relA.selectdirection = "any";
        relB.selectdirection = "any";
      }}
      Plotly.relayout(plotA, relA);
      Plotly.relayout(plotB, relB);
      syncPaintCanvasPointerEvents();
      if (m !== "paint") {{
        clearPaintCanvas(paintCanvasA);
        clearPaintCanvas(paintCanvasB);
      }}
    }}

    function handlePlotSelected(ev, plotGd) {{
      const mode = getPlotDragMode();
      if (mode !== "select") return;
      const pts = ev && ev.points ? ev.points : [];
      const fromBase = pts.filter((p) => p.curveNumber === 0);
      if (fromBase.length === 0) {{
        clearPlotlySelectionChromeBoth();
        return;
      }}
      const idxList = fromBase.map((p) => p.pointIndex);
      applyRegionSelectionByIndices(plotGd, plotGd === plotA, idxList);
    }}

    document.querySelectorAll('input[name="plot-drag-mode"]').forEach((r) => {{
      r.addEventListener("change", applyPlotDragMode);
    }});

    plotA.on("plotly_selected", (ev) => handlePlotSelected(ev, plotA));
    plotB.on("plotly_selected", (ev) => handlePlotSelected(ev, plotB));
    plotA.on("plotly_deselect", () => clearPlotlySelectionChromeBoth());
    plotB.on("plotly_deselect", () => clearPlotlySelectionChromeBoth());

    document.addEventListener("mousedown", (ev) => {{
      const mode = getPlotDragMode();
      if (mode !== "select" && mode !== "paint") return;
      if (!plotPanelA || !plotPanelB) return;
      const el = ev.target;
      if (el.closest(".toolbar")) return;
      if (el.closest(".cat-table-wrap")) return;
      if (plotPanelA.contains(el) || plotPanelB.contains(el)) return;
      clearPlotlySelectionChromeBoth();
    }});

    let paintStroke = null;

    function redrawPaintStroke() {{
      if (!paintStroke) return;
      const {{ canvas, ptsClient }} = paintStroke;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const rc = canvas.getBoundingClientRect();
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(255,198,88,0.95)";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i < ptsClient.length; i++) {{
        const p = ptsClient[i];
        const lx = p.x - rc.left;
        const ly = p.y - rc.top;
        if (i === 0) ctx.moveTo(lx, ly);
        else ctx.lineTo(lx, ly);
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function attachPaintCanvas(canvas, gd, fromPlotA) {{
      function onDocMove(ev) {{
        if (!paintStroke || paintStroke.canvas !== canvas) return;
        if ((ev.buttons & 1) === 0) return;
        paintStroke.ptsClient.push({{ x: ev.clientX, y: ev.clientY }});
        redrawPaintStroke();
      }}
      function onDocUp(ev) {{
        document.removeEventListener("mousemove", onDocMove);
        document.removeEventListener("mouseup", onDocUp);
        if (!paintStroke || paintStroke.canvas !== canvas) return;
        const stroke = paintStroke;
        paintStroke = null;
        const {{ gd: gdv, fromPlotA: fA, ptsClient }} = stroke;
        if (ptsClient.length < 4) {{
          clearPaintCanvas(canvas);
          return;
        }}
        const poly = [];
        for (const p of ptsClient) {{
          const d = screenToDataXY(gdv, p.x, p.y);
          if (d) poly.push(d);
        }}
        if (poly.length < 3) {{
          clearPaintCanvas(canvas);
          return;
        }}
        poly.push({{ ...poly[0] }});
        const ptsArr = fA ? aPts : bPts;
        const hitIdx = [];
        for (let i = 0; i < ptsArr.length; i++) {{
          const q = ptsArr[i];
          if (pointInPolygon(q.x, q.y, poly)) hitIdx.push(i);
        }}
        clearPaintCanvas(canvas);
        applyRegionSelectionByIndices(gdv, fA, hitIdx);
      }}
      canvas.addEventListener("mousedown", (ev) => {{
        if (getPlotDragMode() !== "paint") return;
        ev.preventDefault();
        paintStroke = {{
          canvas,
          gd,
          fromPlotA,
          ptsClient: [{{ x: ev.clientX, y: ev.clientY }}],
        }};
        redrawPaintStroke();
        document.addEventListener("mousemove", onDocMove);
        document.addEventListener("mouseup", onDocUp);
      }});
    }}

    attachPaintCanvas(paintCanvasA, plotA, true);
    attachPaintCanvas(paintCanvasB, plotB, false);

    plotA.on("plotly_afterplot", () => {{
      if (getPlotDragMode() === "paint") resizePaintCanvases();
    }});
    plotB.on("plotly_afterplot", () => {{
      if (getPlotDragMode() === "paint") resizePaintCanvases();
    }});

    window.addEventListener("resize", () => {{
      if (getPlotDragMode() === "paint") resizePaintCanvases();
    }});

    requestAnimationFrame(() => {{
      resizePaintCanvases();
      syncPaintCanvasPointerEvents();
    }});

    updateLineSubsetShapes();
    renderCategoryMetrics("");
    rebuildUnifiedPairsTable();
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def build_data(args) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    texts = _load_text_batch(args.dataset_name, args.dataset_split, args.dataset_text_key, args.num_samples)

    tokenizer_a = AutoTokenizer.from_pretrained(args.model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(args.model_b_name)
    inputs_a = _get_inputs(tokenizer_a, texts, args.max_length)
    inputs_b = _get_inputs(tokenizer_b, texts, args.max_length)

    model_a = _load_model(args.model_a_name, device)
    model_b = _load_model(args.model_b_name, device)

    weights_a, acts_a_2d, acts_a_3d = get_sae_actvs(
        model=model_a,
        model_name=args.model_a_name,
        sae_name=args.sae_a_name,
        inputs=inputs_a,
        layer_id=args.layer_a,
        batch_size=args.batch_size,
        sae_lib="eleuther",
    )
    weights_b, acts_b_2d, acts_b_3d = get_sae_actvs(
        model=model_b,
        model_name=args.model_b_name,
        sae_name=args.sae_b_name,
        inputs=inputs_b,
        layer_id=args.layer_b,
        batch_size=args.batch_size,
        sae_lib="eleuther",
    )

    pool_a = _feature_pool_from_activity(acts_a_2d, args.corr_pool_size)
    pool_b = _feature_pool_from_activity(acts_b_2d, args.corr_pool_size)

    C = build_activation_corr_matrix(
        acts_a_2d,
        pool_a,
        acts_b_2d,
        pool_b,
        batch_cols=max(32, args.corr_batch_size),
    )
    b_to_a_map, b_to_a_corr, a_to_b_map, ot_plan = apply_mapping_method(
        C, pool_a, pool_b, args.mapping_method, args.ot_reg
    )

    b_scores_pool = acts_b_2d[:, pool_b].abs().max(dim=0).values.cpu().numpy()
    selected_b = _select_feature_sets(pool_b, b_scores_pool, args.features_per_side)
    selected_a = np.unique([b_to_a_map[int(feat_b)] for feat_b in selected_b.tolist() if int(feat_b) in b_to_a_map])
    if len(selected_a) == 0 or len(selected_b) == 0:
        raise RuntimeError("No mapped features were selected. Increase --corr-pool-size or provide more input samples.")

    # Prefer, for each selected A, the selected B with highest corr among those mapping to that A (stable UI links).
    a_to_b_map_refined: Dict[int, int] = {}
    for fa in selected_a.tolist():
        best_b: Optional[int] = None
        best_c = float("-inf")
        for fb in selected_b.tolist():
            if b_to_a_map.get(int(fb)) != int(fa):
                continue
            c = float(b_to_a_corr.get(int(fb), float("-inf")))
            if c > best_c:
                best_c = c
                best_b = int(fb)
        if best_b is not None:
            a_to_b_map_refined[int(fa)] = best_b
    a_to_b_map = a_to_b_map_refined

    metrics = compute_alignment_and_metrics(
        acts_a_2d,
        acts_b_2d,
        pool_a,
        pool_b,
        C,
        selected_b,
        b_to_a_map,
        args.mapping_method,
        args.ot_reg,
        ot_plan,
        weights_a,
        weights_b,
    )

    emb_a = _compute_umap(weights_a[selected_a], seed=args.seed)
    emb_b = _compute_umap(weights_b[selected_b], seed=args.seed)

    labels_a = {
        int(idx): _top_token_label(
            acts_a_3d, int(idx), inputs_a["input_ids"].cpu(), tokenizer_a, args.label_top_k
        )
        for idx in selected_a.tolist()
    }
    labels_b = {
        int(idx): _top_token_label(
            acts_b_3d, int(idx), inputs_b["input_ids"].cpu(), tokenizer_b, args.label_top_k
        )
        for idx in selected_b.tolist()
    }

    payload = _build_payload(
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        sae_a_name=args.sae_a_name,
        sae_b_name=args.sae_b_name,
        layer_a=args.layer_a,
        layer_b=args.layer_b,
        selected_a=selected_a,
        selected_b=selected_b,
        emb_a=emb_a,
        emb_b=emb_b,
        b_to_a_map=b_to_a_map,
        b_to_a_corr=b_to_a_corr,
        a_to_b_map=a_to_b_map,
        labels_a=labels_a,
        labels_b=labels_b,
        semantic_categories=_load_semantic_categories(args.semantic_categories_json),
        metrics=metrics,
    )
    for g in payload["semantic_groups"]:
        fa_arr, fb_arr = _pair_arrays_for_semantic_group(
            g, b_to_a_map, a_to_b_map, selected_b, selected_a
        )
        g["metrics"] = compute_subset_metrics(
            acts_a_2d,
            acts_b_2d,
            weights_a,
            weights_b,
            fa_arr,
            fb_arr,
            include_svcca=False,
        )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Build and render a linked SAE feature UMAP map for Pythia models.")
    parser.add_argument(
        "--model-a-name",
        default="EleutherAI/pythia-70m",
        help="Base LM for model A (non-deduped Pythia suite; matches Lan et al. 2025 naming).",
    )
    parser.add_argument(
        "--model-b-name",
        default="EleutherAI/pythia-160m",
        help="Base LM for model B (non-deduped Pythia suite).",
    )
    parser.add_argument("--sae-a-name", default="EleutherAI/sae-pythia-70m-32k")
    parser.add_argument("--sae-b-name", default="EleutherAI/sae-pythia-160m-32k")
    parser.add_argument(
        "--layer-a",
        type=int,
        default=2,
        help="Residual-stream SAE layer for model A. Default 2: strong SVCCA in Fig. 2 (70m vs 160m), Lan et al. 2025.",
    )
    parser.add_argument(
        "--layer-b",
        type=int,
        default=3,
        help="Residual-stream SAE layer for model B. Default 3: pairs with --layer-a=2 as a high-SVCCA cell in Fig. 2.",
    )
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-split", default="train[:256]")
    parser.add_argument("--dataset-text-key", default="text")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--corr-batch-size", type=int, default=128)
    parser.add_argument(
        "--corr-pool-size",
        type=int,
        default=3000,
        help="Number of high-activation features per model to keep before alignment. "
        "Hungarian requires equal pool sizes; use the same value implicitly for both sides.",
    )
    parser.add_argument(
        "--mapping-method",
        choices=["greedy", "hungarian", "ot"],
        default="hungarian",
        help="How to align pool_b columns to pool_a: column-wise argmax (greedy), "
        "global one-to-one max-sum correlation (Hungarian, equal pool sizes), or "
        "entropic OT (Sinkhorn via POT) then argmax per column for display.",
    )
    parser.add_argument(
        "--ot-reg",
        type=float,
        default=0.05,
        help="Sinkhorn entropic regularization for --mapping-method ot (POT).",
    )
    parser.add_argument("--features-per-side", type=int, default=1200)
    parser.add_argument("--label-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-json", default="outputs/pythia_sae_feature_map.json")
    parser.add_argument("--output-html", default="outputs/pythia_sae_feature_map.html")
    parser.add_argument("--title", default="Pythia SAE Feature Space Mapping")
    parser.add_argument("--from-json", default=None, help="If set, skip compute and render directly from a JSON payload.")
    parser.add_argument(
        "--semantic-categories-json",
        default=None,
        help="Path to JSON list of {id, name, keywords: [substr, ...], pattern?: regex string}. "
        "If omitted, a large built-in keyword list (plus optional regex per category) is used; see DEFAULT_SEMANTIC_CATEGORIES in this file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_json = Path(args.data_json)
    output_html = Path(args.output_html)
    wrote_json = False

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        cats = _load_semantic_categories(args.semantic_categories_json)
        prev_by_id = {str(g.get("id")): g for g in (payload.get("semantic_groups") or [])}
        payload["semantic_groups"] = _semantic_groups_for_points(
            payload["model_a"]["points"],
            payload["model_b"]["points"],
            cats,
        )
        for g in payload["semantic_groups"]:
            prev = prev_by_id.get(str(g.get("id")))
            if prev and isinstance(prev.get("metrics"), dict):
                g["metrics"] = prev["metrics"]
    else:
        payload = build_data(args)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload), encoding="utf-8")
        wrote_json = True

    _render_html(payload, output_html=output_html, title=args.title)
    if wrote_json:
        print(f"Saved payload JSON: {output_json.resolve()}")
    print(f"Saved linked UMAP HTML: {output_html.resolve()}")


if __name__ == "__main__":
    main()
