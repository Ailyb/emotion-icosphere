# Judge validation (granite4.2:8b; PLAN_v2 T10 gate: macro-F1 >= 0.5)

GoEmotions frozen set n=840 (manifest f47b22ae79d3...): macro-F1 0.648, accuracy 0.646, unparsed 10. **Gate: PASS**

Confusion (true -> predicted counts):

- joy: {'joy': 67, 'sadness': 2, 'anger': 4, 'surprise': 4, 'love': 4, 'neutral': 10}
- sadness: {'joy': 2, 'sadness': 70, 'anger': 5, 'surprise': 3, 'disgust': 3, 'neutral': 15}
- fear: {'joy': 1, 'sadness': 9, 'fear': 36, 'anger': 5, 'surprise': 6, 'disgust': 4, 'neutral': 1, 'unparsed': 1}
- anger: {'joy': 3, 'sadness': 6, 'anger': 82, 'surprise': 15, 'disgust': 5, 'love': 1, 'neutral': 11, 'unparsed': 5}
- surprise: {'joy': 4, 'sadness': 1, 'anger': 3, 'surprise': 57, 'disgust': 3, 'love': 1, 'neutral': 15}
- disgust: {'joy': 1, 'sadness': 10, 'fear': 1, 'anger': 12, 'surprise': 9, 'disgust': 35, 'love': 1, 'neutral': 5}
- love: {'joy': 15, 'sadness': 3, 'anger': 5, 'surprise': 2, 'disgust': 2, 'love': 117, 'neutral': 6, 'unparsed': 1}
- neutral: {'joy': 8, 'sadness': 11, 'fear': 1, 'anger': 21, 'surprise': 17, 'disgust': 5, 'love': 6, 'neutral': 79, 'unparsed': 3}
