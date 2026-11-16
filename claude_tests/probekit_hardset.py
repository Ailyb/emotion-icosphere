#!/usr/bin/env python3
"""
probekit_hardset.py -- writes data/hardset/hardset.jsonl (47 hand-written passages) and an own-voice template.

Subtypes (each defeats a different shortcut):
  implicit              emotion carried by the situation; no emotion words, no body-language clichés
  negation              emotion words appear but NEGATED; the label is the emotion actually present
  emotion_word_neutral  emotion words as names/topics in neutral text; the label is neutral
  plain_neutral         ordinary neutral text with no emotion vocabulary at all
  own_voice             YOUR passages (see below)

Caveat: these were written by one author (the assistant), so they share one voice, and labels for implicit
passages are judgment calls. 47 passages is small: read results with the confidence intervals the report prints.

OWN VOICE: fill the "text" fields in data/hardset/own_voice_TEMPLATE.jsonl (2-4 sentences each, your own style;
don't name the emotion, avoid clichés like 'heart racing'; mix in some negation and some understatement), then
save as data/hardset/own_voice.jsonl. Rows with empty text are ignored.
"""
import json
from pathlib import Path

H = [
    # ---- implicit -------------------------------------------------------------------------------------
    ("joy", "implicit", "The letter said the loan had been approved. I read it standing in the hallway, then went back and read it again, and then I called my sister and told her to book the tickets."),
    ("joy", "implicit", "When the results came in, Marcus got up from his desk and walked the whole length of the office to tell each person himself, even the ones he barely knew."),
    ("joy", "implicit", "She hadn't expected the whole team to stay late for her send-off. She kept refilling everyone's glass and wouldn't let anyone leave before one more story."),
    ("sadness", "implicit", "He kept her coffee mug on the shelf where it had always been. Each morning he washed it, dried it, and put it back without ever using it."),
    ("sadness", "implicit", "After the funeral the house was full of casseroles nobody ate. I sat in the car in the driveway for an hour before I could make myself go inside."),
    ("sadness", "implicit", "Every year on the fifteenth she still set out two plates, then put one back in the cupboard before anyone came in."),
    ("fear", "implicit", "The footsteps upstairs stopped when I stopped. I turned off the lamp, slid the phone under the blanket, and listened to see whether they would start again."),
    ("fear", "implicit", "She checked the lock a third time, then moved the chair under the handle and sat facing the door until it got light."),
    ("fear", "implicit", "The doctor's office said the doctor wanted to speak to me in person and wouldn't say more over the phone. I replayed that sentence all the way home and couldn't make it mean anything good."),
    ("anger", "implicit", "He said it was a misunderstanding. I laid the three emails on the table in order, with the dates, and asked him to explain each one in a level voice."),
    ("anger", "implicit", "She typed the reply, deleted it, and typed a shorter, colder one. Then she copied the entire department on it."),
    ("anger", "implicit", "The third time the landlord promised to fix the heating and didn't, I stopped calling and started writing down every date, every name, and every excuse for the housing board."),
    ("surprise", "implicit", "I opened the file to fix a typo and found my own name on the winners list. I checked the date, then the spelling, then the date again."),
    ("surprise", "implicit", "My brother, who had said he couldn't get time off, was standing at the gate with a sign. I walked right past him before it registered."),
    ("surprise", "implicit", "The quiet new neighbor turned out to be the author of the book I'd been recommending to everyone for years. I just stood there holding the mail."),
    ("disgust", "implicit", "I lifted the lid and it reached me before I saw it. I set the container down at arm's length and went to open every window."),
    ("disgust", "implicit", "He licked his thumb and turned the page of my book, then did it again, leaving a wet smear in the margin. I took the book back and wiped it on my sleeve."),
    ("disgust", "implicit", "The sink hadn't been cleaned in what looked like weeks. I used a fork to push the plate aside and washed my hands twice before I touched the food."),
    ("calm", "implicit", "The deadline was in an hour and nothing was finished, but he made tea, opened the window, and worked through the list one item at a time."),
    ("calm", "implicit", "She sat on the dock with her feet in the water while the light changed. There was nowhere to be and nothing she needed to decide."),
    ("calm", "implicit", "The train was cancelled. I bought a coffee, found a bench, and watched the pigeons until the next one was announced, in no hurry at all."),
    ("love", "implicit", "He learned to braid her hair from videos, badly, and she wore it to school anyway, touching it all day."),
    ("love", "implicit", "She drove two hours in the rain to bring him the soup he liked, then sat on the edge of his bed until he fell asleep."),
    ("love", "implicit", "After forty years he still saved her the crusty end of the loaf, and she still pretended not to notice."),
    # ---- negation (label = the emotion actually present) --------------------------------------------------
    ("calm", "negation", "I wasn't afraid at all. I walked into the dark room, found the switch, and sat down with my book."),
    ("joy", "negation", "People kept asking if I was sad about leaving. I wasn't sad in the least; I was already planning what to pack for the best year of my life."),
    ("calm", "negation", "He apologized, and I wasn't angry. I nodded, finished my sandwich, and asked what time we should leave."),
    ("sadness", "negation", "I'm not angry with her, and I never was. I just keep setting the table for three out of habit."),
    ("fear", "negation", "Everyone says I should be excited about the surgery. I can't stop checking the clock and rereading the consent form."),
    ("surprise", "negation", "It wasn't scary, only completely unexpected. The whole crowd had hidden in the garden, and I just stood there with my keys in my hand."),
    ("love", "negation", "I didn't say I love you; I never do. I just ironed his shirts every Sunday for fifty years and left the porch light on."),
    ("disgust", "negation", "It didn't taste bad, exactly. But I had seen how it was made, so I thanked her and left the plate untouched."),
    ("anger", "negation", "I didn't raise my voice. I stood up, closed my laptop, and told him the contract was over, effective today."),
    # ---- emotion words in neutral contexts (label = neutral) ------------------------------------------------
    ("neutral", "emotion_word_neutral", "The Fear and Loathing exhibit opens at the library on Thursday. Tickets are available at the front desk until five."),
    ("neutral", "emotion_word_neutral", "The anger management seminar for new supervisors is in Room 204 at three o'clock. Please bring your badge."),
    ("neutral", "emotion_word_neutral", "Joy Dental updated its opening hours: Monday to Friday, eight to four. The office is closed on public holidays."),
    ("neutral", "emotion_word_neutral", "The report classifies customer sentiment as sad, happy, or neutral, and the model reached ninety percent accuracy on the validation set."),
    ("neutral", "emotion_word_neutral", "Chapter four covers disgust as a basic emotion in psychology, including cross-cultural studies of facial expressions."),
    ("neutral", "emotion_word_neutral", "Surprise packages are delivered to the loading dock between nine and ten, and the driver needs a signature."),
    ("neutral", "emotion_word_neutral", "The calm before the storm is a common idiom; the glossary entry lists its origin and three example sentences."),
    ("neutral", "emotion_word_neutral", "Love Street runs north from the station, and the bus stops at the corner of Love Street and Fifth Avenue every twelve minutes."),
    # ---- plain neutral -----------------------------------------------------------------------------------------
    ("neutral", "plain_neutral", "The warehouse receives shipments on Tuesdays and Fridays. Staff log each pallet, check the count against the manifest, and move the boxes to the correct aisle."),
    ("neutral", "plain_neutral", "He took the 7:40 bus to the office, answered emails until noon, and then reviewed the quarterly numbers with the finance team."),
    ("neutral", "plain_neutral", "To reset the router, hold the small button on the back for ten seconds and wait until the front light turns solid green."),
    ("neutral", "plain_neutral", "The recipe calls for two cups of flour, a teaspoon of salt, and enough water to form a dough that holds together."),
    ("neutral", "plain_neutral", "The town council approved the revised parking rules on a vote of five to two and scheduled the next meeting for the second Monday of the month."),
    ("neutral", "plain_neutral", "She packed the samples into the case, labeled each tube, and carried it to the lab before the courier arrived."),
]


def main(out_dir="data/hardset"):
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"hard|{i}", "category": c, "subtype": s, "text": t} for i, (c, s, t) in enumerate(H)]
    (d / "hardset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cats = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "calm", "love"]
    tpl = [{"category": c, "subtype": "own_voice", "text": ""} for c in cats for _ in range(5)]
    tpl += [{"category": "neutral", "subtype": "own_voice", "text": ""} for _ in range(5)]
    (d / "own_voice_TEMPLATE.jsonl").write_text("\n".join(json.dumps(r) for r in tpl) + "\n")
    print(f"wrote {len(rows)} hand-written passages -> {d/'hardset.jsonl'}")
    print(f"wrote {len(tpl)} blank slots (5 per emotion + 5 neutral) -> {d/'own_voice_TEMPLATE.jsonl'}")
    print("Fill in 'text', save as own_voice.jsonl.")


if __name__ == "__main__":
    main()
