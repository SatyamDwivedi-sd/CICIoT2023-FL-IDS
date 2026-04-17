"""
Project-wide constants: label taxonomies, class encodings, feature metadata.

All mapping logic that was duplicated across notebooks lives here.
Import from this module everywhere; never redeclare these in scripts.
"""

# ---------------------------------------------------------------------------
# Fine-grained (34-class) label → 8-category string
# ---------------------------------------------------------------------------
CATEGORY_MAP: dict[str, str] = {
    "BENIGN": "Benign",
    # DDoS
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    # DoS
    "DOS-UDP_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",
    # Mirai
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    # Recon
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",
    "VULNERABILITYSCAN": "Recon",
    # Spoofing
    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",
    # Web
    "XSS": "Web",
    "COMMANDINJECTION": "Web",
    "BACKDOOR_MALWARE": "Web",
    "SQLINJECTION": "Web",
    "UPLOADING_ATTACK": "Web",
    "BROWSERHIJACKING": "Web",
    # BruteForce
    "DICTIONARYBRUTEFORCE": "BruteForce",
}

# ---------------------------------------------------------------------------
# Attack taxonomy: category → list of fine-grained labels
# (inverse of CATEGORY_MAP, grouped)
# ---------------------------------------------------------------------------
ATTACK_TAXONOMY: dict[str, list[str]] = {
    "DDoS": [
        "DDOS-ICMP_FLOOD", "DDOS-UDP_FLOOD", "DDOS-TCP_FLOOD",
        "DDOS-PSHACK_FLOOD", "DDOS-SYN_FLOOD", "DDOS-RSTFINFLOOD",
        "DDOS-SYNONYMOUSIP_FLOOD", "DDOS-ACK_FRAGMENTATION",
        "DDOS-UDP_FRAGMENTATION", "DDOS-ICMP_FRAGMENTATION",
        "DDOS-SLOWLORIS", "DDOS-HTTP_FLOOD",
    ],
    "DoS": ["DOS-UDP_FLOOD", "DOS-TCP_FLOOD", "DOS-SYN_FLOOD", "DOS-HTTP_FLOOD"],
    "Mirai": ["MIRAI-GREETH_FLOOD", "MIRAI-UDPPLAIN", "MIRAI-GREIP_FLOOD"],
    "Recon": [
        "RECON-HOSTDISCOVERY", "RECON-OSSCAN", "RECON-PORTSCAN",
        "RECON-PINGSWEEP", "VULNERABILITYSCAN",
    ],
    "Spoofing": ["MITM-ARPSPOOFING", "DNS_SPOOFING"],
    "Web": [
        "XSS", "COMMANDINJECTION", "BACKDOOR_MALWARE",
        "SQLINJECTION", "UPLOADING_ATTACK", "BROWSERHIJACKING",
    ],
    "BruteForce": ["DICTIONARYBRUTEFORCE"],
    "Benign": ["BENIGN"],
}

# ---------------------------------------------------------------------------
# 8-category integer encoding (order matches notebook 06/07/08)
# ---------------------------------------------------------------------------
CATEGORY_ENCODING: dict[str, int] = {
    "DDoS": 0,
    "DoS": 1,
    "Mirai": 2,
    "Benign": 3,
    "Recon": 4,
    "Spoofing": 5,
    "Web": 6,
    "BruteForce": 7,
}
CATEGORY_NAMES: dict[int, str] = {v: k for k, v in CATEGORY_ENCODING.items()}

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
NUM_FEATURES: int = 39
NUM_BINARY_CLASSES: int = 2
NUM_CAT8_CLASSES: int = 8
NUM_FINE34_CLASSES: int = 34

# Balancing targets used in notebook 06
UNDER_SAMPLING_TARGETS: dict[int, int] = {
    CATEGORY_ENCODING["DDoS"]: 500_000,
    CATEGORY_ENCODING["DoS"]: 500_000,
    CATEGORY_ENCODING["Mirai"]: 500_000,
    CATEGORY_ENCODING["Benign"]: 500_000,
    CATEGORY_ENCODING["Recon"]: 200_000,
    CATEGORY_ENCODING["Spoofing"]: 200_000,
}

OVER_SAMPLING_TARGETS: dict[int, int] = {
    CATEGORY_ENCODING["Web"]: 100_000,
    CATEGORY_ENCODING["BruteForce"]: 50_000,
}
