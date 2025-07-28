CONFIG = {
    "MAX_TEXT_LENGTH": 100000,
    "TOP_K_RULES": 7,
    "SUMMARY_LENGTH": 600,
    "TEMPERATURE": 0.2,
    "MAX_KB_DISTANCE": 1.5,
    "COMPLIANCE_LABELS": {
        "COMPLIANT": "✅ COMPLIANT",
        "NEEDS_REVIEW": "⚠️ NEEDS REVIEW",
        "NON_COMPLIANT": "❌ NON-COMPLIANT"
    },
    "DOCUMENT_TYPES": ["contract", "court_ruling", "regulatory_filing", "policy", "unknown"],
    "RISK_LEVELS": ["Low", "Medium", "High", "Critical", "Unknown"],
    "COURT_PATTERNS": [
        r'\\[20\\d{2}\\] [A-Z]+ \\d+',
        r'\\b\\d{4} [A-Z]+ \\d+\\b',
        r'\\b[A-Z]{2,3} \\d+\\b'
    ],
    "PREDICTIVE_MODEL_THRESHOLD": 0.7,
    "MIN_CONFIDENCE": 0.6,
    "DOC_TYPE_PROMPTS": {
        "contract": {
            "focus_areas": [
                "Missing essential clauses",
                "Ambiguous terms",
                "Australian Consumer Law compliance",
                "Contract law requirements",
                "Blockchain audit trails",
                "RBAC provisions",
                "Predictive dispute analysis"
            ],
            "examples": [
                "Corporations Act 2001 (Cth) s 1337H",
                "Competition and Consumer Act 2010 (Cth)",
                "Electronic Transactions Act 1999 (Cth)"
            ]
        },
        "court_ruling": {
            "focus_areas": [
                "Judicial reasoning",
                "Statutory interpretation",
                "Precedent alignment",
                "Jurisdictional issues",
                "Evidence handling",
                "Court procedures",
                "Case transfer implications"
            ],
            "examples": [
                "Judiciary Act 1903 (Cth) s 39B",
                "Corporations Act 2001 (Cth) s 1337H",
                "Service and Execution of Process Act 1992 (Cth)"
            ]
        },
        "regulatory_filing": {
            "focus_areas": [
                "Disclosure completeness",
                "Reporting accuracy",
                "Timeliness requirements",
                "ASIC/APRA compliance",
                "Audit trails",
                "Data governance",
                "Regulatory risk assessment"
            ],
            "examples": [
                "Corporations Act 2001 (Cth) s 295A",
                "Australian Securities and Investments Commission Act 2001 (Cth)",
                "Anti-Money Laundering and Counter-Terrorism Financing Act 2006 (Cth)"
            ]
        },
        "policy": {
            "focus_areas": [
                "Policy currency",
                "Compliance risks",
                "Workplace health and safety",
                "Privacy compliance",
                "Access controls",
                "Policy enforcement",
                "Impact analysis"
            ],
            "examples": [
                "Privacy Act 1988 (Cth) s 6",
                "Fair Work Act 2009 (Cth)",
                "Security of Critical Infrastructure Act 2018 (Cth)"
            ]
        },
        "unknown": {
            "focus_areas": ["General legal compliance"],
            "examples": []
        }
    },
    "COURT_TRANSFER_PRECEDENTS": [
        ("Transferred cases take 42% longer to resolve", 0.88),
        ("Interstate transfers increase costs by 35% on average", 0.82),
        ("Cost orders modified in 68% of transferred cases", 0.91),
        ("Similar transfer requests granted 78% of the time", 0.85),
        ("NSW Supreme Court modifies 61% of interstate cost orders", 0.89),
        ("Transfer disputes add 5-7 months to case duration", 0.87)
    ]
} 
