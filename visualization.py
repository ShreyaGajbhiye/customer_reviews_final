import nltk
import matplotlib
matplotlib.use('Agg')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# # Ensure NLTK data packages are downloaded
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return words

def lemmatize_text(text, ngram_range=(1,2)):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    custom_stop_words = set([
        'customer', 'mention', 'enhance', 'service', 'improve', 'ensure',
        'provide', 'experience', 'business', 'review', 'reviews', 'feedback',
        'product', 'us', 'also', 'many', 'one', 'could', 'would', 'may',
        'well', 'much', 'even', 'get', 'go', 'say', 'take', 'make', 'come',
        'think', 'see', 'lot', 'really', 'good', 'great', 'excellent',
        'food', 'place', 'time', 'order', 'restaurant', 'eat', 'like'
    ])
    stop_words.update(custom_stop_words)

    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    # words = [word for word in words if word not in stop_words]
    words_pos = pos_tag(words)
    
    lemmatized_words = []
    for word, tag in words_pos:
        if tag.startswith('J'):
            pos = 'a'  # Adjective
        elif tag.startswith('V'):
            pos = 'v'  # Verb
        elif tag.startswith('N'):
            pos = 'n'  # Noun
        elif tag.startswith('R'):
            pos = 'r'  # Adverb
        else:
            pos = 'n'  # Default to noun
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=pos))
    lemmatized_words = [word for word in lemmatized_words if word not in stop_words]

    #Generate n-grams
    ngrams_list = []
    for n in range(ngram_range[0],ngram_range[1] +1 ):
        ngrams_list.extend([' '.join(grams) for grams in nltk.ngrams(lemmatized_words, n)])

    return ngrams_list

def red_color_func(*args, **kwargs):
    return "rgb(255, 0, 0)"  # Solid red color

def generate_wordcloud_image(text_list, ngram_range=(1, 1),color_func=None):
    combined_text = ' '.join(text_list)
    lemmatized_words = lemmatize_text(combined_text, ngram_range)
    if not lemmatized_words:
        lemmatized_words = ['No', 'Data']  # To handle empty word lists
    if color_func == 'red':
        current_color_func = red_color_func
    else:
        current_color_func = None
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations = False, max_words=100,color_func=current_color_func).generate(' '.join(lemmatized_words))
    # Save the word cloud to a BytesIO object
    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    # Encode the image to base64 string
    encoded_img = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(encoded_img)



def get_top_keywords(reviews, ngram_range=(1,2), top_n=20):
    """
    Extracts the top N keywords from the given reviews.

    Parameters:
    - reviews (list): A list of review texts.
    - top_n (int): The number of top keywords to return.

    Returns:
    - most_common (list of tuples): A list of tuples containing keywords and their counts.
    """

    lemmatized_words = lemmatize_text(' '.join(reviews), ngram_range)
    word_counts = Counter(lemmatized_words)
    most_common = word_counts.most_common(top_n)
    return most_common

def generate_keyword_bar_chart(keywords):
    words = [word for word, count in keywords]
    counts = [count for word, count in keywords]
    # Normalize counts for color mapping
    counts_array = np.array(counts)
    norm = Normalize(vmin=counts_array.min(), vmax=counts_array.max())
    # Choose a colormap (e.g., 'Blues', 'Reds', 'viridis', 'plasma', 'coolwarm')
    colormap = cm.get_cmap('plasma')
    colors = colormap(norm(counts_array))

    fig, ax = plt.subplots(figsize=(6, 4))
    # Create the horizontal bar chart
    bars = ax.barh(words, counts, color=colors)
    ax.invert_yaxis()
    # Set labels and title
    ax.set_xlabel("Frequency", fontsize=10)
    ax.set_title("Top Keywords in Reviews", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add color bar (optional)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Frequency', rotation=270, labelpad=10)
    plt.tight_layout()

    img = BytesIO()
    fig.savefig(img, format = 'png', bbox_inches = 'tight')
    plt.close(fig)
    img.seek(0)

    encoded_img = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(encoded_img)



####trying to build the LDA model.

def build_lda_model(reviews, num_topics = 5):
    """
    build an lda model and extracts topics
    returns:
    -lda model
    -corpus
    -gensim dictionary object

    """
    ###preprocessing the reviews: a simple tokenizer is required because lda will do the rest of the thing
    preprocess_reviews = [clean_text(review) for review in reviews]
    ##create dictionary and corpus
    dictionary = Dictionary(preprocess_reviews)
    corpus = [dictionary.doc2bow(text) for text in preprocess_reviews]
    lda_model =LdaModel(corpus = corpus, id2word = dictionary, num_topics = num_topics, random_state = 12, passes = 10)
    return lda_model, corpus, dictionary

def extract_topics(lda_model, num_words = 5):
    """
    extract topics and their top words
    num_words = num_words per topic
    """
    topics = lda_model.show_topics(nnum_topics = -1, num_words = num_words, formatted = False)
    return [{"Topic": i, "Words": [word for word, _ in words]}for i, words in topics]

def visualize_topics(lda_model, corpus, dictionary):
    """
    Visualize topics using pyLDAvis

    """
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return pyLDAvis.prepared_data_to_html(vis)


# Visualize topics using pyLDAvis
def visualize_lda(lda_model, corpus, dictionary):
    """
    Visualizes LDA topics using pyLDAvis.

    Parameters:
    - lda_model: The trained LDA model.
    - corpus: Corpus for the LDA model.
    - dictionary: Gensim dictionary object.

    Returns:
    - encoded_img: Base64-encoded image of the LDA visualization.
    """
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return pyLDAvis.prepared_data_to_html(vis)


# Static bar chart for LDA topics
def generate_topic_bar_chart(topics):
    """
    Generates a bar chart for LDA topics.

    Parameters:
    - topics (list): List of topics with their top words.

    Returns:
    - encoded_img: Base64-encoded image of the bar chart.
    """
    topic_labels = [f"Topic {t['Topic']}" for t in topics]
    word_counts = [len(t["Words"]) for t in topics]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(topic_labels, word_counts, color="skyblue")
    ax.set_xlabel("Number of Words", fontsize=10)
    ax.set_ylabel("Topics", fontsize=10)
    ax.set_title("LDA Topics", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()

    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)

    encoded_img = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(encoded_img)


