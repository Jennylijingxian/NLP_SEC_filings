# NLP_SEC_filings
Analysis of SEC 10-K filings  
<br>
**Scope:**  
SEC 10-K files for S&P 500 Companies from 2010-2019, roughly 143G  
<br>
**Contents:**
1. 10-K Parsing  
SEC 10-K files has multiple components: 10-K, exibits, XMLs, etc. So firstly we need to  split the big 10-K file into different parts and then send them to correponding parser to extract natural language.
Our main focus is 10-K itself, and it's in a html format, so we mainly use BeautifulSoup, Regex and NLTK for text extraction.
2. 10-K Analysis and Visualization
  - Document similarity calculation and visualization
  - Tokenized word count change from 2010-2019
  - Wordcloud plot for important words from 2010-2019
  - Components of 10-K files and its change from 2010-2019
3. Word2Vec Embedding
  - Skipgram (with Negative Sampling)
  - CBOW
  - Initialized with GloVe weights.
4. Novel Word Embedding Evaluation: How to evaluate if the embedding have leared the unique relationship between words under financial context?
  - Embeddings-based uncertainty detection: link stock volatility with liguistic uncertainty
  - Bootstrap method with Loughran-McDonald Sentiment Word List, more details at [Presentation2.pptx](./Presentation2.pptx)
5. BERT & FinBert Model for Sentence Classification
