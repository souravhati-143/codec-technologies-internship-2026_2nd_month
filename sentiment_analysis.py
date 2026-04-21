# ============================================================
# Project 3: Sentiment Analysis Tool
# Author: Sourav Hati
# Internship: Codec Technologies - AI Internship 2026
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("       SENTIMENT ANALYSIS TOOL")
print("   By: Sourav Hati | Codec Technologies 2026")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1: Try VADER (best), fallback to manual lexicon
# ─────────────────────────────────────────
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    USE_VADER = True
    print("\n✅ NLTK VADER loaded — using advanced sentiment analysis")
except Exception:
    USE_VADER = False
    print("\n⚠️  NLTK not found — using built-in lexicon (still works!)")

# ─────────────────────────────────────────
# STEP 2: Sample Movie Reviews Dataset
# ─────────────────────────────────────────
reviews = [
    ("This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.", "Positive"),
    ("I loved every minute of this film. Brilliant storytelling and amazing performances.", "Positive"),
    ("What a masterpiece! One of the best movies I have ever seen in my life.", "Positive"),
    ("Great film with wonderful characters. Highly recommend to everyone!", "Positive"),
    ("The cinematography was breathtaking and the music score was outstanding.", "Positive"),
    ("An enjoyable movie with good performances and an interesting story.", "Positive"),
    ("Pretty decent film overall. Some parts were great, others felt a bit slow.", "Neutral"),
    ("The movie was okay. Not the best but not the worst either.", "Neutral"),
    ("Average film with a predictable storyline but decent acting.", "Neutral"),
    ("It was fine. Nothing too special but watchable for a lazy evening.", "Neutral"),
    ("Had some good moments but also some boring parts. Mixed feelings overall.", "Neutral"),
    ("This film was a complete disaster. Terrible acting and a boring plot.", "Negative"),
    ("Worst movie I have seen in years. Total waste of time and money.", "Negative"),
    ("The story made no sense whatsoever. Poorly written and badly directed.", "Negative"),
    ("Absolutely dreadful. I nearly fell asleep halfway through this disaster.", "Negative"),
    ("Disappointing film with weak characters and a confusing storyline.", "Negative"),
    ("The special effects were good but the script was awful and the ending was terrible.", "Negative"),
    ("A heartwarming and inspiring story that touched my soul deeply.", "Positive"),
    ("Not worth watching. Very slow and extremely dull throughout.", "Negative"),
    ("Surprisingly good! Exceeded all my expectations completely.", "Positive"),
]

texts = [r[0] for r in reviews]
true_labels = [r[1] for r in reviews]

print(f"\n📊 Dataset: {len(reviews)} movie reviews loaded")

# ─────────────────────────────────────────
# STEP 3: Sentiment Analysis
# ─────────────────────────────────────────

def manual_sentiment(text):
    """Simple lexicon-based sentiment analyzer"""
    positive_words = set([
        'fantastic','superb','loved','brilliant','amazing','masterpiece',
        'wonderful','great','outstanding','breathtaking','enjoyable','good',
        'heartwarming','inspiring','exceeded','good','superb','excellent',
        'perfect','awesome','incredible','magnificent','beautiful','happy',
        'best','recommend','touching','special','decent','fine'
    ])
    negative_words = set([
        'disaster','terrible','boring','worst','waste','awful','dreadful',
        'disappointing','weak','confusing','slow','dull','bad','poor',
        'horrible','pathetic','stupid','ugly','useless','fails','failure',
        'worst','nonsense','unwatchable','fell asleep','not worth'
    ])
    words = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    if pos > neg + 1:
        return 'Positive', pos / max(len(words), 1)
    elif neg > pos + 1:
        return 'Negative', neg / max(len(words), 1)
    else:
        return 'Neutral', 0.5

predicted_labels = []
confidence_scores = []

if USE_VADER:
    sia = SentimentIntensityAnalyzer()
    for text in texts:
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            predicted_labels.append('Positive')
        elif compound <= -0.05:
            predicted_labels.append('Negative')
        else:
            predicted_labels.append('Neutral')
        confidence_scores.append(abs(compound))
else:
    for text in texts:
        label, conf = manual_sentiment(text)
        predicted_labels.append(label)
        confidence_scores.append(conf)

# ─────────────────────────────────────────
# STEP 4: Accuracy
# ─────────────────────────────────────────
correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
accuracy = correct / len(true_labels) * 100

print(f"\n🤖 MODEL: {'VADER (NLTK)' if USE_VADER else 'Lexicon-based'}")
print(f"🎯 Accuracy: {accuracy:.1f}%")
print(f"\n{'─'*55}")
print(f"{'Review (truncated)':<40} {'True':>8} {'Pred':>8}")
print(f"{'─'*55}")
for i, (text, true, pred) in enumerate(zip(texts, true_labels, predicted_labels)):
    match = "✅" if true == pred else "❌"
    print(f"{match} {text[:38]:<38} {true:>8} {pred:>8}")

# ─────────────────────────────────────────
# STEP 5: Interactive Prediction
# ─────────────────────────────────────────
print(f"\n{'─'*55}")
print("🔮 CUSTOM TEXT PREDICTIONS:")
custom_texts = [
    "This AI internship is absolutely amazing and very educational!",
    "The project deadline is stressing me out badly.",
    "The weather today is neither good nor bad.",
]
for ct in custom_texts:
    if USE_VADER:
        s = sia.polarity_scores(ct)
        lbl = 'Positive' if s['compound'] >= 0.05 else ('Negative' if s['compound'] <= -0.05 else 'Neutral')
        conf = abs(s['compound'])
    else:
        lbl, conf = manual_sentiment(ct)
    emoji = "😊" if lbl == "Positive" else ("😠" if lbl == "Negative" else "😐")
    print(f"\n  Text : {ct[:55]}")
    print(f"  Result: {emoji} {lbl} (confidence: {conf:.2f})")

# ─────────────────────────────────────────
# STEP 6: Visualization
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Sentiment Analysis Tool – Movie Reviews\nSourav Hati | Codec Technologies AI Internship 2026',
             fontsize=13, fontweight='bold')

colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}

# Plot 1: Predicted distribution pie chart
pred_counts = Counter(predicted_labels)
ax1 = axes[0]
wedges, texts_pie, autotexts = ax1.pie(
    pred_counts.values(),
    labels=pred_counts.keys(),
    autopct='%1.1f%%',
    colors=[colors[k] for k in pred_counts.keys()],
    startangle=90,
    textprops={'fontsize': 11}
)
ax1.set_title('Sentiment Distribution\n(Predicted)', fontweight='bold')

# Plot 2: True vs Predicted bar chart
categories = ['Positive', 'Neutral', 'Negative']
true_counts = [true_labels.count(c) for c in categories]
pred_counts_list = [predicted_labels.count(c) for c in categories]
x = np.arange(len(categories))
width = 0.35
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, true_counts, width, label='True', color=['#27ae60','#e67e22','#c0392b'], alpha=0.8)
bars2 = ax2.bar(x + width/2, pred_counts_list, width, label='Predicted', color=['#2ecc71','#f39c12','#e74c3c'], alpha=0.8)
ax2.set_title('True vs Predicted Count', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Count')
ax2.legend()
ax2.set_ylim(0, 10)
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(int(bar.get_height())), ha='center', fontsize=10)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(int(bar.get_height())), ha='center', fontsize=10)

# Plot 3: Confidence scores per review
ax3 = axes[2]
bar_colors = [colors[p] for p in predicted_labels]
ax3.barh(range(len(confidence_scores)), confidence_scores, color=bar_colors, alpha=0.8)
ax3.set_title('Confidence Score per Review', fontweight='bold')
ax3.set_xlabel('Confidence Score')
ax3.set_ylabel('Review #')
ax3.set_xlim(0, 1)
patches = [mpatches.Patch(color=colors[k], label=k) for k in colors]
ax3.legend(handles=patches, loc='lower right', fontsize=9)
ax3.text(0.5, -2, f'Overall Accuracy: {accuracy:.1f}%', ha='center',
         fontsize=11, fontweight='bold',
         transform=ax3.get_yaxis_transform())

plt.tight_layout()
plt.savefig('sentiment_analysis_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved as 'sentiment_analysis_output.png'")
print(f"\n🎉 Project 3 Complete! Accuracy: {accuracy:.1f}% | Ready for GitHub!")