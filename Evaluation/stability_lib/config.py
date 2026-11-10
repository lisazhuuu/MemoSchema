SOFT_MATCH_MIN_OVERLAP = 0.30

# Stopwords stripped from field names during normalization.
STOP_WORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "by", "to",
    "for", "with", "as", "is", "be",
}

# Schema-noise tokens stripped during canonical-key normalization so that
# e.g. ``microglial_activation_state``, ``microglial_activation_status`` and
# ``microglial_activation_level`` all collapse to the same key.
#
# Remove common measurement qualifiers. Keep terms such as "name",
# "class" and "family", which can distinguish concepts.
NOISE_TOKENS = {
    # type / identity indicators
    "type", "types",
    "identity", "identities",
    "info", "information", "data",
    # state / measurement / readout indicators (the main biomedical noise group)
    "state", "states",
    "status", "statuses",
    "level", "levels",
    "value", "values",
    "amount", "amounts",
    "presence",
    "burden", "extent", "degree", "magnitude",
    "load", "loads", "count", "counts",
    "abundance", "concentration",
    "profile", "profiles",
    "change", "direction",
    "response",
    # production-mode synonyms ("X release" / "X production" / "X secretion"
    # / "X expression" / "X output" / "X generation" all describe how much
    # of X there is)
    "production", "release", "secretion", "expression",
    "output", "generation",
}

