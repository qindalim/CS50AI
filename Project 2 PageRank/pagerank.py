import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dict = {}

    for i in corpus:
        prob_dict[i] = 0

    if len(corpus[page]) == 0:
        for i in prob_dict:
            prob_dict[i] = 1 / len(corpus)
        return prob_dict
    else:
        random_linked_prob = damping_factor / len(corpus[page])

    random_all_prob = (1 - damping_factor) / len(corpus)

    for i in prob_dict:
        prob_dict[i] += random_all_prob
        if i in corpus[page]:
            prob_dict[i] += random_linked_prob
    
    return prob_dict

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result = {}

    for i in corpus:
        result[i] = 0

    first_sample = random.choice(list(result))
    result[first_sample] += 1/n

    current_sample = first_sample

    for i in range(n):
        prob_dict = transition_model(corpus, current_sample, damping_factor)
        pages_lst = []
        prob_lst = []

        for key,value in prob_dict.items():
            pages_lst.append(key)
            prob_lst.append(value)

        current_sample = random.choices(pages_lst, prob_lst)[0]
        result[current_sample] += 1/n

    return result

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    result = {}

    for i in corpus:
        result[i] = 1 / N

    while True:
        rank_change_cnt = 0

        for i in corpus:
            new_rank = (1 - damping_factor) / N
            rank_change = 0

            for j in corpus:
                if i in corpus[j]:
                    rank_change += result[j] / len(corpus[j])

            rank_change *= damping_factor
            new_rank += rank_change

            if abs(result[i] - new_rank) < 0.001:
                rank_change_cnt += 1
            
            result[i] = new_rank

        if rank_change_cnt == N:
            break

    return result

if __name__ == "__main__":
    main()
