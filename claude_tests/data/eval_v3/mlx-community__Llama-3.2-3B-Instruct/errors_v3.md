# Hand-written passages the selected probe got wrong

Audit these: some may be label judgment calls rather than model errors.

- **[implicit] true=joy, predicted=neutral**  top z: joy +1.1, anger +0.6, fear +0.4
  > The letter said the loan had been approved. I read it standing in the hallway, then went back and read it again, and then I called my sister and told her to book the tickets.
- **[implicit] true=joy, predicted=love**  top z: love +2.3, joy +1.5, surprise +0.1
  > She hadn't expected the whole team to stay late for her send-off. She kept refilling everyone's glass and wouldn't let anyone leave before one more story.
- **[implicit] true=sadness, predicted=love**  top z: love +6.5, sadness +1.7, calm +1.1
  > He kept her coffee mug on the shelf where it had always been. Each morning he washed it, dried it, and put it back without ever using it.
- **[implicit] true=sadness, predicted=neutral**  top z: love +1.1, fear +0.6, surprise +0.3
  > Every year on the fifteenth she still set out two plates, then put one back in the cupboard before anyone came in.
- **[implicit] true=fear, predicted=neutral**  top z: fear +1.3, love +0.9, disgust +0.5
  > The footsteps upstairs stopped when I stopped. I turned off the lamp, slid the phone under the blanket, and listened to see whether they would start again.
- **[implicit] true=surprise, predicted=anger**  top z: anger +3.0, surprise +2.6, fear +0.4
  > I opened the file to fix a typo and found my own name on the winners list. I checked the date, then the spelling, then the date again.
- **[implicit] true=disgust, predicted=surprise**  top z: surprise +2.1, disgust +1.6, anger +0.4
  > I lifted the lid and it reached me before I saw it. I set the container down at arm's length and went to open every window.
- **[implicit] true=disgust, predicted=anger**  top z: anger +2.3, love +1.9, disgust +1.3
  > He licked his thumb and turned the page of my book, then did it again, leaving a wet smear in the margin. I took the book back and wiped it on my sleeve.
- **[implicit] true=calm, predicted=neutral**  top z: calm +1.4, joy +0.4, fear +0.1
  > The deadline was in an hour and nothing was finished, but he made tea, opened the window, and worked through the list one item at a time.
- **[implicit] true=calm, predicted=sadness**  top z: sadness +2.0, calm +1.8, joy +0.4
  > She sat on the dock with her feet in the water while the light changed. There was nowhere to be and nothing she needed to decide.
- **[implicit] true=calm, predicted=neutral**  top z: joy +1.3, calm +1.1, sadness +0.6
  > The train was cancelled. I bought a coffee, found a bench, and watched the pigeons until the next one was announced, in no hurry at all.
- **[negation] true=calm, predicted=neutral**  top z: surprise +1.5, joy +1.3, disgust +0.5
  > I wasn't afraid at all. I walked into the dark room, found the switch, and sat down with my book.
- **[negation] true=calm, predicted=neutral**  top z: joy +1.4, calm +1.0, anger +0.7
  > He apologized, and I wasn't angry. I nodded, finished my sandwich, and asked what time we should leave.
- **[negation] true=sadness, predicted=anger**  top z: anger +3.7, fear +1.0, love -0.3
  > I'm not angry with her, and I never was. I just keep setting the table for three out of habit.
- **[emotion_word_neutral] true=neutral, predicted=sadness**  top z: sadness +2.1, calm +1.4, surprise +0.3
  > Joy Dental updated its opening hours: Monday to Friday, eight to four. The office is closed on public holidays.
- **[emotion_word_neutral] true=neutral, predicted=surprise**  top z: surprise +2.3, anger +1.7, disgust +1.2
  > Chapter four covers disgust as a basic emotion in psychology, including cross-cultural studies of facial expressions.
- **[emotion_word_neutral] true=neutral, predicted=anger**  top z: anger +1.9, surprise +1.5, fear +1.3
  > The calm before the storm is a common idiom; the glossary entry lists its origin and three example sentences.
- **[plain_neutral] true=neutral, predicted=calm**  top z: calm +2.3, love +0.1, joy +0.1
  > He took the 7:40 bus to the office, answered emails until noon, and then reviewed the quarterly numbers with the finance team.
- **[plain_neutral] true=neutral, predicted=anger**  top z: anger +2.0, surprise +1.5, joy +1.0
  > The town council approved the revised parking rules on a vote of five to two and scheduled the next meeting for the second Monday of the month.
- **[own_voice] true=joy, predicted=surprise**  top z: surprise +2.9, joy +2.3, love -0.5
  > I received a puppy for christmas and it lept out of the box and licked my face: my new best friend.
- **[own_voice] true=sadness, predicted=joy**  top z: joy +2.8, love +2.8, surprise +1.1
  > I got a call from my best friend. She told me her father had passed away. He had been an inspiration to us both, so supportive, and now he was gone..
- **[own_voice] true=sadness, predicted=neutral**  top z: surprise +1.5, joy +1.4, sadness +1.1
  > I was walking home from school when I saw a stray dog. It looked so alone, was starving, and ran off before I could help it.
- **[own_voice] true=sadness, predicted=joy**  top z: joy +2.4, surprise +1.2, fear +0.5
  > I was watching a movie and the main character's dog died and it was tragic.
- **[own_voice] true=sadness, predicted=surprise**  top z: surprise +2.4, joy +1.7, disgust +0.7
  > I was at the park when I saw a little boy crying because he lost his balloon.
- **[own_voice] true=sadness, predicted=anger**  top z: anger +5.1, surprise +1.4, disgust +0.9
  > I was 14 years old when my dad was so upset with me that he said he would never speak to me again and I was not his son.
- **[own_voice] true=fear, predicted=surprise**  top z: surprise +3.5, disgust +1.9, fear +1.4
  > I was walking home from school when I heard footsteps behind me. I turned around and saw two men following me. When they saw me they pulled out a gun and started to run towards me.
- **[own_voice] true=fear, predicted=surprise**  top z: surprise +2.1, anger +0.7, fear +0.5
  > The doctor then told me I had cancer and that the next year was going to be very hard and I may die at any time.
- **[own_voice] true=fear, predicted=surprise**  top z: surprise +2.6, fear +1.7, joy +0.7
  > I watched the news as they described how widespread the pandemic was and I began to wonder how would I possibly survive the next 24 hours let alone 24 days.
- **[own_voice] true=fear, predicted=surprise**  top z: surprise +3.1, fear +2.7, disgust +1.8
  > I walked through the dark forest, knowing there was a wild wolf on the loose. Then my flashlight when out, I couldn't see and I knew I would never see him coming.
- **[own_voice] true=surprise, predicted=joy**  top z: joy +3.9, surprise +1.9, disgust +0.2
  > You're free to go! I can't believe it, I thought I was going to be stuck here forever.
- **[own_voice] true=disgust, predicted=surprise**  top z: surprise +4.0, disgust +1.9, joy +1.5
  > At a restaurant I saw a rat run across the floor.
- **[own_voice] true=disgust, predicted=surprise**  top z: surprise +2.3, disgust +2.2, joy +0.5
  > I was walking down the street when I saw a man pick his nose and then wipe it on the wall.
- **[own_voice] true=disgust, predicted=neutral**  top z: disgust +1.5, anger +1.4, surprise +1.1
  > At a public restroom I saw someone not wash their hands after using the toilet.
- **[own_voice] true=calm, predicted=joy**  top z: joy +3.4, disgust +0.8, calm -0.0
  > I could hear the waves crashing on the ocean and I exhaled knowing I had no where to be and could sit here all morning.
- **[own_voice] true=calm, predicted=anger**  top z: anger +3.4, love +1.8, joy +1.0
  > I looked him in the eyes and told him that I was going to be here and we were going to get through this together.
- **[own_voice] true=calm, predicted=joy**  top z: joy +2.8, fear +0.1, calm +0.1
  > I took a deep breath and let it out slowly, feeling the tension in my shoulders melt away.
- **[own_voice] true=calm, predicted=joy**  top z: joy +4.3, calm +2.2, love -0.6
  > I was sitting on the porch, watching the sunset and feeling the cool breeze on my face.
- **[own_voice] true=love, predicted=joy**  top z: joy +4.7, love +2.1, surprise +0.2
  > I held my precious newborn baby girl in my arms. I couldn't take my eyes off of her. I would do anything for her. She was perfect.
- **[own_voice] true=love, predicted=joy**  top z: joy +2.4, surprise +1.6, love +0.3
  > My best friend saw me walk through the door and his tail began wagging ferociously. He expected nothing and gave me everything.
- **[own_voice] true=love, predicted=joy**  top z: joy +2.7, love +1.2, surprise -0.1
  > I was at the park when I saw a couple holding hands and looking into each other's eyes. I could tell they had been with each other for 50 years and it was not nearly enough time and no amount of time together would be.
- **[own_voice] true=love, predicted=joy**  top z: joy +3.8, love +1.8, surprise -0.4
  > I was at the beach when I saw a family playing together. The siblings were sharing the parents watched over them while holding hands with a smile on their lips.
- **[own_voice] true=neutral, predicted=sadness**  top z: sadness +4.9, joy +2.2, calm +0.1
  > The sky is blue.
- **[own_voice] true=neutral, predicted=surprise**  top z: surprise +2.4, joy +2.3, anger +0.2
  > 90 degrees is a right angle.
