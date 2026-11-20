532 passages, 32 topics; topics appearing in more than one split: 0

[8-way, neutral excluded (v1-comparable)] topic-disjoint, seeds [0, 1, 2]: per-seed macro-F1 [0.945, 0.929, 0.919], mean 0.931, bootstrap 95% CI [0.906, 0.953] (pooled test n=431, 32 topics; chance 0.125)
GATE (upper CI < 0.7): FAIL

[9-way, neutral included] topic-disjoint, seeds [0, 1, 2]: per-seed macro-F1 [0.938, 0.918, 0.916], mean 0.924, bootstrap 95% CI [0.898, 0.945] (pooled test n=491, 32 topics; chance 0.111)

Top 20 TF-IDF features per class (* = matches the emotion-word filter):
  anger     clenched*, fists*, fuming*, burning, frustration*, kidding, swear, indignation, clenched fists*, fierce, stuck, dare, intensity, smug, face, oblivious, infuriating*, jaw, seething*, supposed
  calm      gentle, soothing*, soft, slow, quiet, hum, stillness*, tea, outside, unhurried*, soft hum, seep, deliberate, pace, time, letting, background, measured, warmth, sipping
  disgust   stomach*, revulsion*, stench*, churn, stomach churn*, stale, smell, making, queasy*, skin, mouth, stomach churning*, making stomach*, protest, crawl, churning, skin crawl, putrid, trying, churned
  fear      heart, racing, heart racing*, making, chest, palms, feel heart, like jackhammer*, jackhammer*, try, racing like, feel, tension, sweat, shaking, throat, breath, pounding*, shadows, chest like
  joy       infectious, excitement*, laughter, floating, warm, music, carefree, feet, feeling, face, buoyant, danced, alive, bouncing, felt, grin, couldn help, help, energy, hair
  love      gentle, hand, way, warmth, smile, presence, gentle pressure, heart, intertwined, sense, hold, chest, connection, pressure, instinctively, comforting, fingers, familiar, way heart, soft
  neutral   ll, area, scheduled, destination, today, minutes, expected, nearby, went, yesterday, location, discuss, event, let know, new, didn, afternoon, stopping, update, pretty
  sadness   weight, emptiness, reminder, hollow, ache*, lost, silence, heavy, blankly, worn, echo, eyes, staring blankly, faint, muster energy, tears*, stark, feeling, muster, like
  surprise  sudden, suddenly, stared, caught, standing, stumbled, wrap head, process, breath caught, frozen, wrap, trying process, trying, mind, jolt, trying wrap, breath, open, just, widened

Most common 2-word openers per class (share of that class):
  anger     'as she' 0.19, 'are you' 0.103, 'as i' 0.086
  calm      'as i' 0.429, 'as she' 0.143, 'as the' 0.102
  disgust   'as she' 0.246, 'i just' 0.193, 'as i' 0.105
  fear      'as she' 0.175, 'as i' 0.159, 'as the' 0.095
  joy       'i just' 0.254, 'as i' 0.254, 'as she' 0.22
  love      'as i' 0.527, 'as she' 0.164, 'as they' 0.073
  neutral   'today i' 0.062, 'i attended' 0.062, 'i've been' 0.047
  sadness   'as i' 0.156, 'as she' 0.141, 'i sit' 0.047
  surprise  'as she' 0.238, 'i just' 0.206, 'i was' 0.111
