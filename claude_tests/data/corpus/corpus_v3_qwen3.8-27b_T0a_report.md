# T0a report: qwen3.8:27b corpus (generated 2026-11-16 with the OLD clean(); dialogue/text-message formatting flattened)

Reproduce: `python probekit_tfidf_gate.py --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl`, `python probekit_corpus_audit.py --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl --log <run.log> --seed 0`, `python probekit_overlap.py --test <hand-written files> --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl`.

## TF-IDF gate (exit status 1 = FAIL: upper CI >= 0.7)
```
1002 passages, 56 topics; topics appearing in more than one split: 0

[8-way, neutral excluded (v1-comparable)] topic-disjoint, seeds [0, 1, 2]: per-seed macro-F1 [0.723, 0.705, 0.695], mean 0.708, bootstrap 95% CI [0.678, 0.741] (pooled test n=809, 56 topics; chance 0.125)
GATE (upper CI < 0.7): FAIL

[9-way, neutral included] topic-disjoint, seeds [0, 1, 2]: per-seed macro-F1 [0.741, 0.722, 0.733], mean 0.732, bootstrap 95% CI [0.706, 0.761] (pooled test n=911, 56 topics; chance 0.111)

Top 20 TF-IDF features per class (* = matches the emotion-word filter):
  anger     sharp, manager, deliberate, number, pulled phone, straight, walked, timestamp, phone, line, chat, list, voice, type, explain, picked, office, damage, apology, message
  calm      let, slow, tea, rush, sip, rhythm, urgency, cup, nice, hurry, coffee, simply, water, sam, rushed, afternoon, second, page, quiet, took
  disgust   air, smell, exit, eyes, leave, distance, open, far, tight, like, scent, don, turned, door, walk, away, smells, stepping, scanned, hallway
  fear      refreshing, times, checking, time, checked, kept, eyes, convinced, seconds, scanning, clock, waiting, refusing, counting, hasn, hovering, twice, despite, staring, fixed
  joy       actually, letting, like, lingered, light, moment, felt, smiling, perfectly, laughed, stop smiling, felt like, exactly, instead, smiled, years, usual, home, immediately, wide
  love      ensuring, knowing, mentioned, noticed, extra, spare, wouldn, ensure, quietly, just case, saved, low, left, case, kept, elena, voice low, blanket, morning, liked
  neutral   yes, scheduled, noted, mark, returned, confirmed, placed, reviewed, checked, pm, current, margaret, ll, 15, saturday, minutes, sarah, task, tasks, received
  sadness   sit, silence, used, tracing, okay, noise, dark, cold, avoiding, longer, haven, asked, think, hey, chair, wall, louder, static, want, said voice
  surprise  stared, mid, staring, thought, supposed, hovering, slowly, expecting, did, froze, wait, realizing, years, entire, childhood, stopped, set, holding, suddenly, mom

Most common 2-word openers per class (share of that class):
  anger     'did you' 0.045, 'mara paused' 0.027, 'marcus set' 0.027
  calm      'i let' 0.055, 'did you' 0.037, 'take your' 0.018
  disgust   'the smell' 0.062, 'i can't' 0.027, 'marcus shifted' 0.027
  fear      'i keep' 0.08, 'marcus kept' 0.054, 'mara kept' 0.054
  joy       'i just' 0.055, 'did you' 0.046, 'i can't' 0.037
  love      'did you' 0.08, 'i noticed' 0.027, 'i kept' 0.027
  neutral   'the man' 0.036, 'i placed' 0.027, 'marcus checked' 0.027
  sadness   'did you' 0.036, 'i left' 0.027, 'i chose' 0.027
  surprise  'i had' 0.107, 'did you' 0.045, 'i thought' 0.036
```

## Overlap with the test-only sets (aggregates only)
```
test passages: 92 (from 2 file(s)); aggregates only, no text printed
data/corpus/corpus_v3_qwen3.8-27b.jsonl: n_corpus=1002, exact=0, containment>=0.2=0, containment>=0.5=0, containment>=0.8=0, max_containment=0.095, any_shared_5gram=4
```

# Audit
## Passages per (category, split): n=1002

| category | test | train | val | total |
|---|---|---|---|---|
| anger | 20 | 72 | 20 | 112 |
| calm | 20 | 69 | 20 | 109 |
| disgust | 20 | 72 | 20 | 112 |
| fear | 20 | 72 | 20 | 112 |
| joy | 20 | 70 | 19 | 109 |
| love | 20 | 72 | 20 | 112 |
| neutral | 20 | 72 | 20 | 112 |
| sadness | 20 | 72 | 20 | 112 |
| surprise | 20 | 72 | 20 | 112 |

## Rejection reasons

During generation (from log): {'word:calm': 18, 'word:unhurried': 17, 'word:stillness': 17, 'word:fear': 14, 'too_long': 10, 'word:cheer': 8, 'word:unexpected': 7, 'word:surpris': 7, 'word:knot': 6, 'word:relax': 6}
Saved passages re-checked with the current filter: {None: 1002} (None = passes)

## Format adherence by requested style

| style | n | has quotes | has line breaks |
|---|---|---|---|
| dry, understated first-person account | 166 | 0.14 | 0.00 |
| plain report-style account where the feeling shows only in w | 170 | 0.12 | 0.00 |
| reflective paragraph in which the writer reasons about what  | 167 | 0.16 | 0.00 |
| short dialogue between two people (use quotation marks) | 166 | 0.99 | 0.00 |
| text-message exchange between two people | 169 | 0.21 | 0.00 |
| third-person narrative focused on one person's actions and c | 164 | 0.25 | 0.00 |

## Length (words) per category

| category | mean | min | max |
|---|---|---|---|
| anger | 100 | 38 | 153 |
| calm | 90 | 39 | 138 |
| disgust | 94 | 28 | 159 |
| fear | 106 | 53 | 161 |
| joy | 97 | 26 | 148 |
| love | 109 | 51 | 176 |
| neutral | 57 | 22 | 109 |
| sadness | 101 | 27 | 170 |
| surprise | 98 | 36 | 156 |

Duplicate ids: 0; duplicate texts: 0

## Random samples (3 per category; judge off-target by reading)

**[anger] a concert crowd / reflective paragraph in which the writer / test**
> The bass thumped against my ribs, but I stopped dancing and scanned the sea of faces, ignoring the urge to blend in with the rhythm. Someone had elbowed me hard enough to spill my drink, and the stain on my shirt was a cold, dark spot that refused to fade into the haze of the strobe lights. I pushed through the throng, parting bodies with deliberate, sharp movements, letting the music become background noise. I wasn’t looking for an apology; I was looking for the specific angle of a shoulder and the quick, guilty look of someone who knew they had broken a boundary. The crowd’s energy felt like a wall I had to break through to find the source of that sudden, silent rupture in the night.

**[anger] a broken laptop / dry, understated first-person account / val**
> The screen is cracked in a spiderweb pattern that started right where I had left my coffee mug last Tuesday, though I was certain I had moved it to the kitchen counter. I opened the lid slowly, watching the pixels fail to align, and noticed a faint, dark ring on the wooden desk that did not match the shape of my cup. I pulled out my phone to call the person who had been in the office that evening, my voice flat as I asked them to step out of the break room for a moment. I did not ask if they remembered leaving their mug on my desk; I simply waited for them to explain why my device was now a piece of modern art.

**[anger] a flea market / dry, understated first-person account / val**
> I stood by the card table, holding the $50 bill I’d just handed over for a watch that had been missing its hands for the past six years. The vendor, a man in a stained polo, didn’t look up from his cigarette as he slid the empty box toward me. I tucked the bill into my pocket and stepped back, scanning the row of folding chairs until my eyes settled on the red plastic one he had occupied an hour ago. I sat down in it, not because I was tired, but because it was directly in his line of sight, and I pulled out my phone to find the number of the consumer protection office.

**[calm] a restaurant kitchen / dry, understated first-person account / test**
> I let the timer on the sous-vide run an extra three minutes, knowing the steak would still be fine. When the line cook asked if I needed a hand with the garnish, I shook my head and kept my focus on the current plate. The clatter of pans and the shouting from the pass felt like distant background noise, irrelevant to the task at hand. I took my time wiping down the station, choosing a clean cloth rather than the one already in use. There was no rush to move on to the next station, and I made sure not to look at the clock.

**[calm] a family dinner / reflective paragraph in which the writer / train**
> He let the salt shaker sit untouched on the table, deciding that the broth was sufficient as it was. While his sister debated the merits of ordering a second course, he simply refilled his water glass and listened to the hum of the refrigerator from across the kitchen. He picked up a breadstick, tore it in half with a steady hand, and set the other piece back on the plate without eating it. There was no rush to answer when his father asked about the weekend; he just nodded, watched the candle flicker slightly in the draft from the open window, and continued to stir his soup in slow, circular motions.

**[calm] a group project / short dialogue between two people (use q / train**
> Did you finish the slides for the final presentation?" "Not yet, but I’m just finishing up a coffee." "We have the meeting in twenty minutes." "I’ll be there when I’m done; the internet is slow today, so I’m taking my time." "You’re going to get us in trouble." "Probably, but the view from the window is nice, so I’ll stay a bit longer.

**[disgust] a bus ride home / plain report-style account where the fee / train**
> Marcus shifted his weight to the far edge of the bench, pressing his shoulder against the window glass instead of facing the aisle. He kept his eyes fixed on the rearview mirror, deliberately ignoring the woman beside him who was loudly recounting a family dinner to her phone. When the bus lurched forward, he did not steady himself by touching her arm; instead, he gripped the overhead rail with white-knuckled force, leaning into the metal frame to create a physical barrier. He checked the digital display every thirty seconds, his thumb hovering over the "Exit" button as if waiting for permission to leave. At the next stop, he was the first one off, moving faster than usual to clear the crowded platform and head straight for the quieter, dimmer end of the station.

**[disgust] a town council meeting / short dialogue between two people (use q / test**
> Are you serious about installing the neon light installation in the main square?" Sarah asked, adjusting her glasses as if the spectacle were a visual glitch. She kept her gaze fixed on the agenda’s font size, refusing to look up at the proposed renderings on the screen. "I think I will step out for some fresh air before the next speaker takes the podium." "It’s just a lighting project," Mark replied, oblivious to her tension, "it’s meant to be uplifting." "Uplifting my desire to find the nearest exit, perhaps," Sarah muttered, already turning her chair toward the window and away from the glowing blue hues reflecting on the table.

**[disgust] a new roommate / text-message exchange between two people / val**
> I’m already packing my bags, the smell of mildew from their shoe collection is seeping into the walls and I can’t scrub it out. I tried to open the windows to air the place out, but the draft just pushes the odor toward my desk where I spend all my time. I’ve stopped eating in the kitchen because the steam from the rice cooker carries that damp, forgotten scent straight into the dining area. It feels less like sharing a home and more like being slowly encased in something alive and suffocating. I’m checking out tomorrow night, I just need to survive until the moving truck arrives.

**[fear] a parking ticket / dry, understated first-person account / val**
> The envelope in my pocket feels like it has gained weight since I passed the mailman’s window three blocks ago. I’ve stopped looking at the street numbers and started checking my rearview mirror every time I approach an intersection. My thumb has smoothed the flap of the envelope until the paper is soft, but the crease remains. I left the headlights on while I sat in the driveway for ten minutes, just to avoid the silence of the house.

**[fear] a community garden / text-message exchange between two people / test**
> I saw the council meeting minutes posted on the community board this morning. They’re proposing to rezone the east lot for commercial development by next month. I’ve already drafted three different objection letters, each one more formal than the last. I can’t sleep, so I’m just sitting here refreshing the website to see if the comment period has closed yet. You’re not answering your phone, so I assume you’re still pretending this doesn’t affect our plot.

**[fear] a late-night phone call / reflective paragraph in which the writer / train**
> The clock on the nightstand reads 2:17 AM, a time that feels less like a point on a dial and more like an open wound. I keep checking the phone screen, waiting for a name that does not appear, while the silence in the room grows heavy and suffocating. I try to close the eyes, but the mind circles the worst-case scenarios with mechanical precision, listing every way things could unravel. I stand up, walk to the window, and press my forehead against the cold glass, watching the streetlamps flicker, wondering if I should just drive over there now or wait for the dawn.

**[joy] a bus ride home / plain report-style account where the fee / train**
> The passenger taps the exit button a full two stops early, ignoring the automated chime. Instead of glancing at their phone, they press their face against the cool glass, watching the familiar streetlamps flicker on one by one. They straighten their posture, check their watch, and take an extra, deliberate sip of cold coffee from their thermos. As the doors slide open, they step out onto the pavement with a brisk stride, already dialing a number before their feet have cleared the curb.

**[joy] a broken laptop / short dialogue between two people (use q / val**
> I thought I lost three years of research when the screen cracked, but the technician managed to recover every single file from the backup drive before the end of the day." "You saved my career." "I noticed you didn't say 'thank you' like you usually do; you just grabbed a cup of coffee and sat back down at your desk with a wide smile." "The relief of seeing my documents intact made me want to hum while I type, which is something I haven't done since college.

**[joy] a late delivery / reflective paragraph in which the writer / train**
> The package arrived three days late, wrapped in a tape that had clearly seen better days, yet the moment I peeled it open, I realized the contents were exactly what I had hoped for. I immediately set aside my phone to ignore the pending customer service email and instead arranged the items on the kitchen table with meticulous care. A warm light from the afternoon sun hit the surface, highlighting the craftsmanship I had only seen in pictures, so I spent the next hour simply turning the object over in my hands. I decided not to call the courier or file a complaint; instead, I poured a fresh cup of coffee and sat down to examine the details that would take weeks to replicate. The silence in the room felt less like emptiness and more like a held breath before the next chapter began.

**[love] a garden in spring / reflective paragraph in which the writer / train**
> Elias paused before sweeping the fallen branches, noticing that the path to the neighbor’s porch was already clear of debris, yet a single, stubborn dandelion still rooted itself near her threshold. He decided to leave it, knowing she liked to sketch wildflowers, and instead spent the afternoon moving the heavy oak planter box two feet closer to her window, positioning it just so the morning light would catch the soil. When she called out that the coffee was on, he didn’t answer, but he made sure to water the ferns she had neglected for the week, checking the moisture level with his thumb as if it were a pulse.

**[love] a bus ride home / reflective paragraph in which the writer / train**
> The driver announces the next stop, and I check my phone to ensure my sister has arrived safely at the restaurant before I let myself move toward the exit. I linger by the window, watching the streetlights blur into streaks of gold, knowing she likely forgot her umbrella in this unpredictable weather. I pull out my own compact one from my bag, the handle worn smooth from years of use, and decide to call her again just to confirm. I remain seated, letting the rhythm of the bus rock me forward, because rushing would only make my hands shake when I dial, and she doesn't need to hear that in my voice tonight.

**[love] a hiking trip / dry, understated first-person account / train**
> I left my own lunch in the pack and kept the extra sandwich and water bottle for you, just in case. When we reached the overlook, I shifted my position to block the wind and adjusted the angle of the shade so the sun didn’t hit your back. I noticed you checking the time twice and said there was no rush to reach the trailhead by dark, offering to shorten the route if the light began to fail. You asked if I was cold, and I told you I was fine, while silently pulling your jacket zipper up two inches higher.

**[neutral] a group project / plain report-style account where the fee / train**
> The group convened to divide the research tasks among its members. One member collected the sources while another drafted the outline. They met twice a week to compare notes. The final document was submitted on time. No further meetings were scheduled.

**[neutral] a hiking trip / third-person narrative focused on one pe / train**
> Mark laced his boots and checked the water levels in his two bottles before stepping onto the gravel path. He followed the marked trail for four miles, taking a ten-minute break at the midpoint to refill his container from a stream. He noted the time on his watch and resumed walking without stopping to admire the scenery. Upon reaching the parking lot at dusk, he packed his gear into the car and drove home.

**[neutral] a flea market / plain report-style account where the fee / val**
> He sorted the items into three piles on the folding table: sell, keep, and donate. The lamp with the frayed cord was placed in the sell pile because the repair cost exceeded its value. He packed the remaining boxes with newspaper and taped them shut. At noon, he loaded the car and drove to the recycling center without stopping for lunch.

**[sadness] a neighbor's dog / text-message exchange between two people / train**
> Mara: It’s howling again at 3 AM. I heard it through the floorboards. Sam: Yeah, Mr. Whiskers is a nightmare. Did you sleep? Mara: No. I just kept staring at the wall until the sun came up. Sam: You okay? Mara: I’m fine. Just trying to decide if I should move the couch or just leave it where he used to sit.

**[sadness] a first day at a new job / third-person narrative focused on one pe / train**
> Mara slid into the assigned chair, leaving it slightly too far from the desk, and stared at the blinking cursor on her screen without typing a single character. She accepted the complimentary coffee offered by the receptionist but left it untouched, the paper cup growing cold and damp on her knee while she watched other colleagues engage in animated conversations she could barely hear. When a coworker asked about her weekend, she offered a monosyllabic nod and a faint, vacant smile, her eyes remaining fixed on a water stain on the ceiling far above the group. She did not look up when the supervisor walked by, nor did she adjust her posture; she simply remained a still, unresponsive presence in a room full of motion.

**[sadness] a moving truck / dry, understated first-person account / val**
> The mover lifts the heavy oak dresser and slides it into the back of the truck without asking for permission. I stand by the open tailgate, watching the wood scrape against the metal floor, and say nothing when he asks if there is anything else for that room. My hands remain in my pockets, fingers idly tracing the same crease in my denim jeans over and over again. When the truck’s engine sputters to life, I do not look up to check if the tailgate latch has clicked into place.

**[surprise] a birthday party / plain report-style account where the fee / train**
> Marcus stood frozen by the kitchen island, his hand still hovering over the slice of cake he had just cut, while he stared at the small, framed photograph sitting on the center table that he had not seen before. He set his fork down precisely and straightened the photo’s angle by two degrees, ensuring it faced the window rather than the guests. After a long silence, he turned to his wife and asked, "Where did you get that?", his voice flat and steady. He then picked up his plate, left the remaining cake on the table, and walked quietly to the back door, opening it to the cool evening air instead of joining the singing.

**[surprise] an unexpected rainstorm / plain report-style account where the fee / train**
> Marcus dropped his umbrella into the gutter without a second thought. He scanned the crowd for a dry doorway, ignoring the sleek coat he had packed specifically for this trip. The plan to meet his date at the outdoor cafe was abandoned immediately, replaced by a quick text that read, "I can’t make it." He turned his back to the downpour and walked toward the nearest subway entrance, leaving his shoes to soak on the pavement.

**[surprise] a visit to the vet / text-message exchange between two people / train**
> Oh good, Dr. Evans just said Buster is healthy and can go home. Wait, why is the cat carrier in the back? I only brought the dog. Is he okay? No, wait, that’s not Buster’s carrier.


## Reader's judgment of the 27 samples above
Clearly off-target: surprise/'an unexpected rainstorm' (reads as a cancelled plan); weak: joy/'a bus ride home' (report style), calm/'a group project' dialogue (reads as procrastination). About 2-3 of 27 (7-11%); n=3 per category cannot support a per-category 20% test. Format: dialogue passages lack their outer quotes and all passages are single-paragraph (old clean()).
