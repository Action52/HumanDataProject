CLASS_MAPPER = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}

BIG_SEGMENTS = 10000 # constant for 0s segments in the beginning

SEGMENT_PADDING = 0

LABELS_WITHOUT_ZERO = {
    0: 'Left hand',
    1: 'Right hand',
    2: 'Passive/neutral',
    3: 'Left leg',
    4: 'Tongue',
    5: 'Right leg'
}

LABELS_WITH_ZERO = {
    0: 'idle',
    1: 'Left hand',
    2: 'Right hand',
    3: 'Passive/neutral',
    4: 'Left leg',
    5: 'Tongue',
    6: 'Right leg'
}

ZERO = 0
