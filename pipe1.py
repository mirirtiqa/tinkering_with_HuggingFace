from transformers import pipeline

def getlabel(labels,scores):
    maxind =0
    maxscore = scores[0]
    for i in range(1,len(scores)):
        if scores[i] > maxscore:
            maxind = i
            maxscore = scores[i]
    return labels[maxind], maxscore



classifier = pipeline(model="facebook/bart-large-mnli")
text = input("type you sentence that you want classified: ")
res = classifier(
    text,
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
label, prob = getlabel(res['labels'],res['scores'])
print(f'the sentence can classified as {label} with a certainty of {prob}')
