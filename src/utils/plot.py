from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plotWordFrequency(token_prob):
    wc = WordCloud(background_color="white", max_words=10)
    wc.generate_from_frequencies(token_prob)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
