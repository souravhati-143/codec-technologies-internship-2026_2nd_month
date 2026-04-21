# ============================================================
# Project 4: Text Summarizer
# Author: Sourav Hati
# Internship: Codec Technologies - AI Internship 2026
# ============================================================

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("          AI TEXT SUMMARIZER")
print("   By: Sourav Hati | Codec Technologies 2026")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1: Extractive Summarization Engine
# ─────────────────────────────────────────

STOP_WORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','shall',
    'this','that','these','those','i','you','he','she','it','we','they',
    'my','your','his','her','its','our','their','me','him','us','them',
    'what','which','who','whom','how','when','where','why','not','no','so',
    'as','by','from','up','about','into','through','during','before','after',
    'if','then','than','too','very','just','also','there','here','can'
])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def tokenize_words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def get_word_frequencies(text):
    words = [w for w in tokenize_words(text) if w not in STOP_WORDS]
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1
    return {word: count/max_freq for word, count in freq.items()}

def score_sentences(sentences, word_freq):
    scores = {}
    for i, sentence in enumerate(sentences):
        words = tokenize_words(sentence)
        score = sum(word_freq.get(w, 0) for w in words if w not in STOP_WORDS)
        scores[i] = score / max(len(words), 1)
    return scores

def summarize(text, num_sentences=3):
    text = clean_text(text)
    sentences = tokenize_sentences(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences), sentences, {i: 1.0 for i in range(len(sentences))}
    word_freq = get_word_frequencies(text)
    scores = score_sentences(sentences, word_freq)
    top_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    top_indices_sorted = sorted(top_indices)
    summary = ' '.join(sentences[i] for i in top_indices_sorted)
    return summary, sentences, scores

# ─────────────────────────────────────────
# STEP 2: Sample Texts
# ─────────────────────────────────────────
sample_texts = {
    "🤖 Artificial Intelligence": """
    Artificial intelligence is the simulation of human intelligence processes by computer systems.
    These processes include learning, reasoning, and self-correction. AI has become one of the most
    transformative technologies of the 21st century. Machine learning, a subset of AI, enables systems
    to automatically learn and improve from experience without being explicitly programmed. Deep learning
    uses neural networks with many layers to analyze various factors of data. AI is now used in healthcare
    to diagnose diseases, in finance for fraud detection, and in transportation for self-driving vehicles.
    Natural language processing allows machines to understand and respond to human language. Computer vision
    enables machines to interpret and understand visual information from the world. The future of AI holds
    enormous potential for solving complex global challenges in climate, medicine, and education.
    """.strip(),

    "🚀 Space Exploration": """
    Space exploration is the ongoing discovery and exploration of celestial structures in outer space.
    Since the launch of Sputnik in 1957, humans have made tremendous progress in understanding the universe.
    The Apollo missions successfully landed humans on the Moon between 1969 and 1972, marking a historic
    achievement for humanity. Today, organizations like NASA, SpaceX, and ISRO are pushing the boundaries
    of what is possible in space travel. Mars has become the next major target for human exploration,
    with multiple rovers already collecting valuable scientific data on the red planet. Space exploration
    has led to numerous technological innovations that benefit everyday life on Earth, including GPS, weather
    satellites, and advanced medical imaging technology. The James Webb Space Telescope is revealing
    unprecedented details about the early universe and distant galaxies. Private companies are now
    making space tourism a reality for civilian passengers.
    """.strip(),

    "🌿 Climate Change": """
    Climate change refers to long-term shifts in global temperatures and weather patterns.
    While some climate change is natural, scientific evidence shows that human activities have been
    the main driver of climate change since the 1800s. The burning of fossil fuels like coal, oil
    and gas generates greenhouse gas emissions that trap the sun's heat and raise temperatures.
    Rising temperatures are causing glaciers and ice caps to melt, leading to rising sea levels
    that threaten coastal communities worldwide. Extreme weather events like hurricanes, droughts,
    and wildfires are becoming more frequent and severe due to climate change. Renewable energy
    sources such as solar and wind power are critical for reducing carbon emissions globally.
    International agreements like the Paris Climate Accord aim to limit global warming to 1.5 degrees
    Celsius. Individuals can help by reducing energy consumption, eating less meat, and supporting
    sustainable businesses and policies.
    """.strip()
}

# ─────────────────────────────────────────
# STEP 3: Run Summarizer on All Texts
# ─────────────────────────────────────────
results = {}
print("\n" + "─"*55)

for title, text in sample_texts.items():
    summary, sentences, scores = summarize(text, num_sentences=3)
    original_words = len(tokenize_words(text))
    summary_words = len(tokenize_words(summary))
    compression = (1 - summary_words/original_words) * 100
    results[title] = {
        'text': text,
        'summary': summary,
        'sentences': sentences,
        'scores': scores,
        'original_words': original_words,
        'summary_words': summary_words,
        'compression': compression
    }
    print(f"\n📄 {title}")
    print(f"   Original : {original_words} words | {len(sentences)} sentences")
    print(f"   Summary  : {summary_words} words | 3 sentences")
    print(f"   Compression: {compression:.1f}% reduced")
    print(f"\n   📝 SUMMARY:")
    for line in summary.split('. '):
        if line.strip():
            print(f"      • {line.strip()}.")

# ─────────────────────────────────────────
# STEP 4: Custom Text Demo
# ─────────────────────────────────────────
print("\n" + "─"*55)
print("🔮 CUSTOM TEXT DEMO:")
custom = """
Python is a high-level, general-purpose programming language known for its simplicity and readability.
Created by Guido van Rossum and released in 1991, Python has grown to become one of the most popular
programming languages in the world. Its clean syntax makes it an excellent choice for beginners learning
to code. Python is widely used in data science, machine learning, artificial intelligence, web development,
and automation. Libraries like NumPy, Pandas, and Scikit-learn have made Python the dominant language
for data analysis and machine learning tasks. Django and Flask are popular frameworks for building web
applications with Python. The language supports multiple programming paradigms including procedural,
object-oriented, and functional programming. Python has a vast and active community that contributes
thousands of open-source packages to the Python Package Index.
"""
custom_summary, custom_sents, custom_scores = summarize(custom.strip(), num_sentences=2)
print(f"\n   Original: {len(tokenize_words(custom))} words")
print(f"   Summary : {len(tokenize_words(custom_summary))} words")
print(f"\n   📝 SUMMARY:\n   {custom_summary}")

# ─────────────────────────────────────────
# STEP 5: Visualization
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('AI Text Summarizer – Extractive Summarization\nSourav Hati | Codec Technologies AI Internship 2026',
             fontsize=13, fontweight='bold')

topic_names = [t.split(' ', 1)[1] for t in results.keys()]
original_words = [r['original_words'] for r in results.values()]
summary_words = [r['summary_words'] for r in results.values()]
compressions = [r['compression'] for r in results.values()]

# Plot 1: Original vs Summary word count
ax1 = axes[0]
x = np.arange(len(topic_names))
w = 0.35
b1 = ax1.bar(x - w/2, original_words, w, label='Original', color='#3498db', alpha=0.85)
b2 = ax1.bar(x + w/2, summary_words, w, label='Summary', color='#2ecc71', alpha=0.85)
ax1.set_title('Word Count: Original vs Summary', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(topic_names, fontsize=9)
ax1.set_ylabel('Word Count')
ax1.legend()
for bar in b1:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             str(int(bar.get_height())), ha='center', fontsize=9)
for bar in b2:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             str(int(bar.get_height())), ha='center', fontsize=9)

# Plot 2: Compression ratio
ax2 = axes[1]
bar_colors = ['#e74c3c', '#9b59b6', '#f39c12']
bars = ax2.bar(topic_names, compressions, color=bar_colors, alpha=0.85, edgecolor='white')
ax2.set_title('Text Compression Rate (%)', fontweight='bold')
ax2.set_ylabel('Compression %')
ax2.set_ylim(0, 100)
ax2.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='70% target')
ax2.legend()
for bar, comp in zip(bars, compressions):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{comp:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Plot 3: Sentence importance scores for AI text
ax3 = axes[2]
ai_result = list(results.values())[0]
scores_vals = list(ai_result['scores'].values())
sent_labels = [f"S{i+1}" for i in range(len(scores_vals))]
top_n = sorted(ai_result['scores'], key=ai_result['scores'].get, reverse=True)[:3]
bar_cols = ['#2ecc71' if i in top_n else '#bdc3c7' for i in range(len(scores_vals))]
ax3.bar(sent_labels, scores_vals, color=bar_cols, alpha=0.9, edgecolor='white')
ax3.set_title('Sentence Importance Scores\n(AI Topic — Green = Selected)', fontweight='bold')
ax3.set_xlabel('Sentence')
ax3.set_ylabel('Importance Score')
green_patch = mpatches.Patch(color='#2ecc71', label='Selected in summary')
gray_patch = mpatches.Patch(color='#bdc3c7', label='Not selected')
ax3.legend(handles=[green_patch, gray_patch], fontsize=9)

plt.tight_layout()
plt.savefig('text_summarizer_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved as 'text_summarizer_output.png'")
print("\n🎉 Project 4 Complete! Ready for GitHub!")