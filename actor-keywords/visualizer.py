import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_wordcloud(keyword_scores, title="Keyword WordCloud"):
    """
    繪製文字雲
    :param keyword_scores: 包含 (word, score) tuple 的列表
    :param title: 圖表標題
    """
    if not keyword_scores:
        print("No keywords to plot.")
        return

    # 將 tuple list 轉為 dict
    word_score_dict = dict(keyword_scores)
    
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(word_score_dict)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.show()