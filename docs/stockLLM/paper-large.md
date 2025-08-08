Retrieval-augmented Large Language Models for Financial Time Series Forecasting
Mengxi Xiaoa, Zhengyu Chenb, Lingfei Qianc, Zihao Jiangb, Yueru Hec,
Yijing Xud, Yuechen Jiange, Dong Lib, Ruey-Ling Wengc, Jimin Huangf,
Min Penga, Sophia Ananiadouf, Jian-Yun Nieg, Qianqian Xie
∗
a
1
a School of Artificial Intelligence, Wuhan University,
b School of Computer Science, Wuhan University,
c The Fin AI, d Columbia University,
e Stevens Institute of Technology, f University of Manchester,
g University of Montreal
Abstract
Accurately forecasting stock price movements is critical for informed financial decision-making, supporting applications ranging from algorithmic trading to risk management. However, this task remains challenging due to the difficulty of retrieving subtle yet high-impact patterns from noisy financial time-series data, where conventional retrieval methods, whether based on generic language models or simplistic numeric similarity, often fail to capture the intricate temporal dependencies and context-specific signals essential for precise market prediction. To bridge this gap, we introduce FinSrag2
2
Code and data are available at FinSrag .
, the first retrieval-augmented generation (RAG) framework with a novel domain-specific retriever FinSeer for financial time-series forecasting. FinSeer leverages a candidate selection mechanism refined by LLM feedback and a similarity-driven training objective to align queries with historically influential sequences while filtering out financial noise. Such training enables FinSeer to identify the most relevant time-series data segments for downstream forecasting tasks, unlike embedding or distance-based retrieval methods used in existing RAG frameworks. The retrieved patterns are then fed into StockLLM, a 1B-parameter LLM fine-tuned for stock movement prediction, which serves as the generative backbone. Beyond the retrieval method, we enrich the retrieval corpus by curating new datasets that integrate a broader set of financial indicators, capturing previously overlooked market dynamics. Experiments demonstrate that FinSeer outperforms existing textual retrievers and traditional distance-based retrieval approaches in enhancing the prediction accuracy of StockLLM, underscoring the importance of domain-specific retrieval frameworks in handling the complexity of financial time-series data.

1Introduction
Financial time-series forecasting is crucial for maintaining market stability and efficiency, with profound implications for investment decisions, risk assessment, and economic policy formulation (Fama and French, 2000). However, the extreme volatility and nonlinear dynamics of financial markets pose significant analytical challenges, requiring sophisticated approaches to distill actionable signals from complex, noise-laden data streams. Stock movement prediction, which involves determining future price direction (up or down), represents a particularly critical yet demanding task that has attracted substantial research interest (Xie et al., 2023a, 2024a, 2024b). Although recent LLM-based approaches have shown promise in stock prediction, they predominantly rely on textual data like news and social media (Wang et al., 2024), using only the past several days’ closing prices as reference while largely ignoring the wealth of information contained in historical time-series patterns (Bustos and Pomares-Quimbaya, 2020). This oversight creates a critical gap in effectively harnessing temporal financial data for prediction. The challenge is further compounded by the enormous scale and complexity of such data, spanning multiple influencing factors and extended historical contexts. Addressing this requires an intelligent retrieval mechanism capable of efficiently navigating vast time-series datasets to extract and deliver the most relevant patterns, thereby empowering LLMs to generate more accurate and reliable stock movement forecasts.

Despite the critical importance of retrieving relevant historical patterns for stock movement prediction, current embedding-based and distance-based retrieval-augmented generation (RAG) approaches (Joshi et al., 2024; He et al., 2024; Cui et al., 2024) face fundamental limitations when applied to financial time-series data. Conventional embedding-based methods struggle because numeric financial sequences often exhibit superficially similar patterns that lack explicit semantic meaning, making it difficult for text-trained retrievers to distinguish meaningful signals from spurious correlations. Meanwhile, distance-based techniques like Dynamic Time Warping (DTW) (Yang et al., 2024) prove inadequate as they typically rely on single-variable comparisons (e.g., adjusted closing prices), ignoring the rich contextual information from other financial indicators. This approach limits retrieval to simplistic trend matching and fails to capture deeper, context-dependent relationships where sequences with divergent or opposing trends may actually contain complementary predictive signals. The inherent complexity of financial markets demands retrieval methods that can move beyond surface-level pattern matching to identify truly informative historical sequences that may exhibit complex, non-obvious relationships to current market conditions.

Refer to caption
Figure 1:Overview of our time-series RAG framework.
To address these challenges, we introduce FinSrag, the first Financial time-Series RAG framework for stock movement prediction, using FinSeer, a Financial Time-Series Retriever to effectively retrieve the most beneficial historical sequences beyond surface-level pattern matching for downstream forecasting. FinSeer is trained to identify sequences that are embedded with essential or complementary information with limited similarity compared with the query that are beneficial for prediction. A key challenge in training FinSeer lies in the absence of explicit retrieval ground truth, that for any given query, it is inherently ambiguous which historical patterns will most effectively aid forecasting given the vast amount of data. To address this, we propose an LLM-guided relevance estimation, where the language model itself is used to assess and refine the selection of candidate sequences during training. Building on the approach of Zhang et al. (2023), we assess candidate relevance by feeding query-candidate pairs into the LLM and using the generation logits corresponding to the correct answer as a proxy for relevance. Intuitively, candidates yielding higher logits are considered more informative for the forecasting task.

We then train FinSeer to effectively distinguish high-value sequences (positives) from irrelevant sequences (negatives) by learning a retrieval embedding space where relevant candidates are closely aligned with their corresponding queries. Specifically, for each query, FinSeer is optimized to maximize the similarity score between the query embedding and the embedding of top-ranked candidates (i.e., those associated with higher generation logits from the LLM), while simultaneously pushing apart embeddings of low-ranked, uninformative candidates. This contrastive learning objective encourages FinSeer to internalize the LLM’s implicit relevance signals, thereby enhancing its ability to retrieve sequences that are truly beneficial for downstream forecasting.

Beyond improving relevant sequence retrieval, our framework introduces a crucial advancement in what to retrieve by expanding the candidate pool beyond traditional price-based segments. Specifically, we segment the financial time series into candidates, where each candidate corresponds to a time-series segment of a single feature, such as the adjusted close price or a specific technical indicator. Unlike prior approaches that retrieve only from price sequences, for the first time, our method incorporates 28 additional financial indicators, allowing retrieval over a richer set of candidate sequences that capture diverse aspects of market behavior3
3
For instance, candidates derived from the KDJ indicator group, particularly in the overbought region where 
K
>
80
, 
D
>
70
, and 
J
>
90
 can reveal early signals of potential reversals (Wu and Diao, 2015).
. To support this, we construct a new dataset and enriched existing two where time-series segments from both price data and a broad set of financial indicators are included as retrieval candidates, enabling FinSeer to discover deeper predictive patterns beyond surface-level price trends.

Comprehensive evaluations across multiple financial time-series datasets demonstrate that FinSeer, our specialized retriever, is the only approach that consistently enhances LLM forecasting accuracy, outperforming five state-of-the-art retrievers from the MTEB leaderboard when integrated into the FinSrag framework. This superior performance not only validates our domain-specific design for time-series data, but also crucially reveals FinSeer’s unique capability to identify and retrieve the most predictive financial indicator segments. Notably, the distance-based method which cannot leverage FinSrag’s architecture, shows weaker performance, further confirming the advantages of our framework’s structured approach to sequence selection, by incorporating diverse financial indicators as candidates.

More than just improving accuracy, FinSeer’s retrieval patterns provide valuable financial insights: By automatically discovering and leveraging latent market factors that conventional approaches miss, our method offers an empirically grounded alternative to theoretical assumptions in economic models. This data-driven approach enables systematic identification of predictive market features, opening new avenues to study market inefficiencies and dynamic dependencies through the lens of actual predictive relevance rather than ex-ante hypotheses.

In conclusion, our contributions are summarized as follows:

1. We introduce FinSrag, the first RAG framework with domain-specific retriever for financial time-series forecasting. By improving the capability of identifying and retrieving the most relevant time-series data segments, FinSrag significantly enhances LLMs’ forecasting capabilities in financial markets.
2. We propose FinSeer, a novel domain-specific retriever trained via LLM feedback to uncover overlooked market signals. We complement this with an enriched dataset featuring 28 expert-selected technical indicators that go beyond conventional price data.
3. Experimental results demonstrate the effectiveness of our framework, which surpasseses state-of-the-art retrievers and distance-based methods.
2Prelimilaries
Retrieval-augmented financial time-series forecasting (Sezer et al., 2020) involves predicting future values or trends (
G
) based on a given query sequence (
q
) and a set of retrieved historical sequences (
c
). These sequences are collected over time at regular intervals. In the retrieval process, the goal of the retrieval model (
R
) is to efficiently identify and extract the most useful historical sequences from a large pool of candidates. By providing relevant context, the retrieval model enhances the forecasting model’s ability to make accurate and reliable predictions. In the specific task of stock movement prediction (Xu and Cohen, 2018), the problem is framed as a binary classification task: predicting whether a stock’s price will rise or fall on the next trading day. Given a query sequence 
q
, which represents the stock’s price over the previous 
t
 days, the model retrieves relevant sequences as context and predicts the stock’s movement 
M
q
,
d
 for the next trading day 
d
. Details about threshold settings are illustrated in Appendix C.

3Methodology
3.1Retrieval Candidate Pool Design and Dataset Construction
To support our retrieval framework, we introduce a new dataset comprising high-volume stocks from 2022-2023 and enhance two existing datasets with 28 carefully selected financial indicators. Complete dataset statistics are presented in Table 5, while partitioning details and additional specifications appear in Appendix E.1. The following parts detail our methodology for: (1) stock price data collection and indicator selection (in Collection of Indicators), and (2) temporal scope determination and content specification for queries and candidate sequences (in Query and Candidate Construction).

Table 1:Dataset statistics of query and candidate sequence amounts5.
Dataset	License		Trading Dates	Train	Valid	Test
ACL18 (Xu and Cohen, 2018)	MIT License	query	2015.06.03-2015.12.31	3,312	440	2,912
candidate	2014.06.02-2015.12.31	441,694	67,320	444,312
BIGDATA22 (Soun et al., 2022)	Public	query	2020.04.09-2020.12.31	3,229	434	2,868
candidate	2019.04.01-2020.12.31	328,372	44,778	328,372
STOCK23	MIT License	query	2023.01.03-2023.12.31	4,268	570	4,128
candidate	2022.01.03-2023.12.31	404,736	50,592	404,736
 
Collection of Indicators. Our financial data foundation comes from the Yahoo Finance API (Xu and Berkely, 2014), which provides 6 daily trading metrics, including opening, highest, lowest, close prices, adjusted close prices, along with trading volume. While these basic price data offer essential market snapshots, they fail to capture deeper market dynamics crucial for forecasting. For example, if the adjusted close price increases on continuous trading dates, it only shows upward momentum but cannot determine when a trend is exhausted. The KDJ indicator, however, combines price momentum and range position to signal overbought (>80) or oversold (<20) conditions Wu and Diao (2015).

These limitations motivate our multi-layer dataset design to extract deeper and latent market signals beyond raw price movements. This multi-layer feature design is inspired by established practices in quantitative finance and machine learning  Heaton et al. (2017); Jansen (2020); Chen et al. (2024), which emphasize the importance of hierarchical signal extraction from raw market data to enhance model predictive power. The dataset consists of three hierarchical components: (1) Core Price Metrics: We begin with aforementioned 6 fundamental indicators, including raw OHLCV (Open, High, Low, Close, Volume) and Adjusted Close prices. (2) Primary Technical Indicators: To capture mid-level market dynamics (e.g., trend strength, volatility, and price-volume interactions) Lo et al. (2000); Park and Irwin (2007); Wu and Diao (2015), which are not directly observable in raw OHLCV, we incorporate 10 widely-used technical indicators such as log returns, price momentum, and VWAP, whose practice is supported by both empirical studies and industry applications Chan (2013); Fischer and Krauss (2018); Gu et al. (2020a); Ma and Yan (2022). (3) Alpha Factors: To uncover higher-order predictive signals, we identify 18 Alpha Factors via a Mutual Information (MI)-based selection process Guyon and Elisseeff (2003); Jansen (2020). This approach quantifies the nonlinear dependency between candidate features and future returns, allowing us to retain only those indicators with strong predictive relationships. The effectiveness of alpha factors in enhancing financial model performance has been extensively supported in the literature Harvey et al. (2016); Tulchinsky (2019); Gu et al. (2020b); Kakushadze (2016).

The integrated dataset, exemplified in Appendix E.3, combines these three layers (6 price metrics + 10 technical indicators + 18 alpha factors) to provide a multidimensional view of market conditions. Complete specifications for all 34 indicators appear in Table 4.

Query and Candidate Construction. The query construction process involves temporal range determination and query content specification. For temporal boundaries, we establish that each query must have sufficient candidate sequences available for retrieval by requiring that the query date occurs at least one year after the start date of the corresponding dataset split. This temporal buffer ensures adequate historical context for meaningful retrieval. We implement a one-day sliding window across trading days to generate consecutive queries. For the example shown in Figure 2(a), in the ACL18 dataset spanning from 2014-06-02 to 2015-12-31, we define queries as those occurring between 2015-06-03 and 2015-12-31. This configuration guarantees that even the earliest query (2015-06-03) can access a full year of preceding candidate sequences. Each query consists of the stock identifier, query date, and recent market data for query-candidate matching. The market data includes adjusted closing prices from the five most recent trading days. The five-day window aligns with standard financial practices, where multiples of five are commonly used for price change calculations.

The candidate construction follows similar temporal and content specifications as queries. Candidates are drawn from all available historical data preceding each query date, spanning from the dataset’s start date to the trading day before the query. We apply a one-day sliding window to generate sequential candidates, with all stocks in the dataset eligible as candidates regardless of query stock matching. Each candidate consists of the stock ticker, price movement direction on a specific trading date, a relevant financial indicator, and corresponding indicator values calculated from the preceding five consecutive trading days. Figure 2(b) demonstrates this construction through examples of both rising and falling candidates retrieved from the pool.

Our approach enables seamless market monitoring through incremental updates to the candidate pool. When advancing the query date (e.g., from 2015-12-30 to 2015-12-31), we simply append the new trading day’s data (2015-12-30) to the existing pool (2014-06-02 through 2015-12-29), maintaining all historical candidates without requiring model retraining. This efficient update mechanism ensures continuous operation while preserving the complete candidate history.

Refer to caption
Figure 2:Illustration of query and candidate construction. (a) illustrates how to construct all queries for a given dataset. (b) illustrates the corresponding candidate pool for a given query, and the update of candidate pool with trading date updates.
3.1.1Sequence Serialization
Since stock movement prediction depends on the changes in related features rather than their exact values, we serialize stock prices and financial indicators into a time-series format. We use JSON to represent these sequences, as it has been demonstrated to effectively support LLMs in interpreting time-series data (Fang et al., 2024; Singha et al., 2023; Yin et al., 2023).

The following are two examples of a query and a candidate sequence. When inquerying about stock MO on 2015-06-02, the query sequence contains the stock identifier (MO), query date (2015-06-02), and the adjusted close prices of last five trading dates. The serialized sequence is shown below:

{ "query_stock": "MO",
"query_date": "2015-06-02",
"recent_date_list": ["2015-05-26", "2015-05-27", "2015-05-28", "2015-05-29", "2015-06-01"],
"adjusted_close_list": [29.669, 29.9872, 29.8657, 29.6227, 29.6227]}
A potential candidate in the candidate pool represent stock MO on date 2014-07-02, with the highest price as its indicator. The sequence includes the stock ticker (MO), price movement direction (freeze) on a specific trading date (2014-07-02), a key financial indicator (the highest price), and corresponding indicator values calculated from the preceding five consecutive trading days. The serialized sequence is shown below:

{ "candidate_stock": "MO",
"candidate_date": "2014-07-02",
"candidate_movement": "freeze",
"recent_date_list": ["2014-06-25", "2014-06-26", "2014-06-27", "2014-06-30", "2014-07-01"],
"high_list": [42.2, 42.0, 41.86, 42.28, 42.0]}
3.2Retriever Training
We then train FinSeer to effectively distinguish high-value sequences (positives) from irrelevant sequences (negatives) by learning a retrieval embedding space where relevant candidates are closely aligned with their corresponding queries. To achieve this, we score and select positive and negative candidates as the retriever training corpus (in Candidate Scoring and Selection), then specify the training objective (in Training Objective) and conduct knowledge distillation (in Knowledge Distillation).

Candidate Scoring and Selection. To determine whether a candidate sequence assists in predicting the movement of the query, we use LLM feedback to score each candidate. The details of our LLM backbone, StockLLM-1B-Instruct (hereafter referred to as StockLLM), are shown in Appendix F.1. Specifically, for a given query 
q
, we integrate the query sequence and each candidate sequence 
c
i
 from the candidate pool as concurrent inputs to the StockLLM. Then StockLLM outputs logits, which are unnormalized scores representing the model’s confidence for each possible class (e.g., rise or fall). These logits are transformed into probabilities 
P
⁢
(
c
)
 using the softmax function:

P
⁢
(
c
)
=
e
z
c
∑
j
e
z
j
,
(1)
where 
z
c
 is the logit for the correct class (e.g., rise if the true movement is upward) and 
z
j
 represents the logits for all possible classes. The resulting probability 
P
⁢
(
c
)
 serves as the score for the candidate 
c
i
 with respect to the query 
q
.

We rank the candidate sequences in descending order based on their scores 
P
⁢
(
c
)
. The top-1 sequence is selected as a positive candidate, while the bottom 15 sequences are chosen as negative candidates. The sets of selected positive and negative sequences are denoted as 
ℂ
ℙ
 and 
ℂ
ℕ
, respectively.

Training Objective. Our retriever 
R
⁢
(
q
)
 is designed to intelligently distinguish between historically significant sequences 
ℂ
ℙ
 and noisy sequences 
ℂ
ℕ
. The training objective is to ensure that 
R
⁢
(
q
)
 prioritizes sequences from 
ℂ
ℙ
 while minimizing attention to those from 
ℂ
ℕ
. This is achieved by maximizing a similarity measure 
s
⁢
u
⁢
p
⁢
(
q
,
s
)
 between the query sequence 
q
 and candidate sequences 
s
. Mathematically, the retriever’s objective is formulated as:

R
⁢
(
q
)
=
arg
⁡
max
s
∈
ℂ
ℙ
∪
ℂ
ℕ
⁡
s
⁢
u
⁢
p
⁢
(
q
,
s
)
.
(2)
By focusing on sequences that maximize 
s
⁢
u
⁢
p
⁢
(
q
,
s
)
, the retriever ensures that the most informative and contextually relevant historical sequences are identified.

Knowledge Distillation. To leverage the scoring derived from StockLLM, we employ knowledge distillation, which transfers knowledge from the teacher model (StockLLM) to the student model (retriever) by mimicking the teacher’s output distribution. This approach effectively captures nuanced patterns and predictions from StockLLM. Specifically, we minimize the Kullback-Leibler (KL) divergence between the candidate distributions computed using the LLM’s rewards and those predicted by the embedding model. For each query 
q
 and its candidate list 
{
ℂ
ℙ
,
ℂ
ℕ
}
, we derive StockLLM’s rewards for the candidates, denoted as 
{
P
⁢
(
c
i
)
,
i
=
1
,
…
,
n
}
. To make these rewards suitable for distillation, we normalize them using a softmax function with temperature 
α
:

w
i
=
softmax
R
⁢
(
P
⁢
(
c
i
)
α
)
.
(3)
The KL divergence is then computed as follows:

min
⁢
∑
c
−
w
i
×
log
⁡
(
exp
⁡
(
⟨
𝒆
q
,
𝒆
c
i
⟩
/
τ
)
∑
c
′
∈
ℂ
exp
⁡
(
⟨
𝒆
q
,
𝒆
c
′
⟩
/
τ
)
)
,
(4)
where 
𝒆
q
 and 
𝒆
c
i
 are the embeddings of the query 
q
 and candidate 
c
i
, respectively, and 
τ
 is a temperature parameter. This loss function optimizes the similarity between the query embedding and the embeddings of the top-ranked candidates, enhancing the retriever’s ability to accurately predict stock price movements.

3.3Inference
During inference, the key innovation of our FinSrag framework lies in how FinSeer’s retrieval directly enhances StockLLM’s forecasting capability. Given a query, FinSeer first identifies the most relevant historical sequences by evaluating both temporal patterns and predictive relationships learned during training, filtering out noisy but numerically similar candidates that typically mislead traditional retrievers. These selected sequences are then structured and injected into StockLLM’s context window. Crucially, unlike standard RAG that simply concatenates retrieved documents, this end-to-end alignment between retrieval and generation is what enables FinSrag to outperform conventional forecasting pipelines where retrieval and prediction models are optimized separately.

4Experiment
4.1Experimental Settings
Datasets. We evaluate the effectiveness of our RAG framework on the test sets of the three datasets described in Table 5, with ACL18 containing 2,876 query sequences, BIGDATA22 containing 2,868 queries, and STOCK23 containing 4,128 queries. These thousand-scale queries help mitigate random bias, ensuring a robust and reliable evaluation of model performance.

Candidate Pool Settings. To ensure a comprehensive evaluation, for each query sequence, we include all sequences containing financial indicators across all stocks in the test set (not limited to the same stock), with data available up to the query date, as potential candidates. No additional restrictions are imposed, enabling a robust assessment of the models’ performance in real-world financial forecasting scenarios.

Baselines. To evaluate the efficiency of the FinSrag framework, the bare StockLLM-1B-Instruct without retrieval serves as our baseline to figure out whether the retrieval step enhances StockLLM’s prediction ability. To evaluate our retriever FinSeer, we tested other retrieval methods, including random retrieval, DTW distance, and five competitive retrieving models from the top of the MTEB English Leaderboard as baselines, containing: (1) Instructor-large (Su et al., 2023), a 335M instruction-finetuned text embedder that encodes sequences into 768-dimensional tensors. (2) UAE-large-v1 (Li and Li, 2023), a 335M ANGLE-optimized text embedding model that encodes sequences into 1024-dimensional tensors. (3) E5-mistral-7b-instruct (Wang et al., 2023), a 7111M embedder initialized from Mistral-7B-v0.1 (Jiang et al., 2023) and fine-tuned on multilingual datasets, encoding sequences into 4096-di-mensional tensors. (4) BGE-large-en-v1.5 (Xiao et al., 2023), a 335M general embedder pre-trained with RetroMAE (Xiao et al., 2022), encoding sequences into 1024-dimensional tensors. (5) LLM-Embedder (Zhang et al., 2023), a 109M embedder fine-tuned from BGE-large-en-v1.5, designed as a unified embedding model to support diverse retrieval augmentation needs for LLMs. It encodes sequences into 768-dimensional tensors. More details are shown in Appendix F.2.

Evaluation Metrics. We employ Accuracy (ACC) and Matthews Correlation Coefficient (MCC) (Matthews, 1975) to assess the performance of FinSeer and the baseline models on the stock movement prediction task. These metrics evaluate the performance of stock movement prediction based on the distribution of positive and negative samples.

4.2Main Results
As shown in Table 2, experimental results demonstrate our framework’s effectiveness by outperforming all baseline retrieval methods in assisting LLM-based stock movement prediction.

Table 2:Results of stock movement predictions using StockLLM-1B-Instruct and retrieval models.
Retrieving Methods	ACL18		BIGDATA22		STOCK23
(+ StockLLM-1B-Instruct )	ACC	MCC		ACC	MCC		ACC	MCC
w/o Retrieval	0.498	-0.006		0.493	-0.017		0.509	0.021
Random Retrieval	0.485	-0.028		0.495	-0.007		0.496	-0.004
DTW	0.516	0.041		0.500	0.010		0.492	-0.007
Instructor (Su et al., 2023) 	0.498	-0.005		0.493	-0.015		0.505	0.010
UAE (Li and Li, 2023) 	0.486	-0.029		0.493	-0.011		0.494	-0.009
E5 (Wang et al., 2023) 	0.510	0.027		0.498	-0.002		0.499	-0.001
BGE (Xiao et al., 2023) 	0.492	-0.012		0.501	0.002		0.488	-0.014
LLM Embedder (Zhang et al., 2023) 	0.503	0.007		0.459	-0.083		0.503	0.007
FinSeer	0.517	0.035		0.510	0.023		0.542	0.085
 
First, compared with bare StockLLM and random retrieval, FinSeer demonstrates the critical importance of retrieving truly valuable information. While trained on a comprehensive stock movement prediction corpus, bare StockLLM’s limited input with only the recent five-day adjusted close price results in unstable performance that fluctuates around random guessing levels. Similarly, when randomly retrieved sequences are provided as supposedly relevant context, they introduce instability by arbitrarily confusing or occasionally coincidentally benefiting StockLLM’s decision-making.

Second, our comparison with the five top-ranked retrievers from the MTEB English leaderboard reveals FinSeer’s superiority in time-series retrieval. The negligible performance gap between instruction-finetuned retriever (Instructor) and no-retrieval baselines underscores the fundamental challenges of time-series retrieval. Unlike text retrieval, this task cannot rely solely on task understanding since candidate sequences often exhibit visual similarity while differing in predictive value. Other retrievers demonstrate inconsistent cross-dataset performance because their similarity differentiation fails to align with the LLM’s perception of importance. Even LLM Embedder, our backbone model trained with LLM feedback, shows limited generalization to time-series retrieval, further emphasizing the problem’s complexity. Among those retrievers, FinSeer consistently outperforms other retrievers across all datasets, proving its superior ability to learn LLM preferences and effectively enhance time-series forecasting through retrieval.

Third, our framework’s advantages over distance-based retrieval methods highlight the value of incorporating diverse feature types. While DTW achieves comparable performance to FinSeer during specific market periods (ACL18 [2014-2015] and BIGDATA22 [2019-2020]), its retrieval capability proves insufficient during the volatile 2022-2023 period (STOCK23) marked by significant market surges and fluctuations. During this challenging phase, DTW significantly degrades StockLLM’s performance, while FinSeer maintains its enhancement capability.

4.3Ablation Study
In this section, we explore two aspects of our framework based on retrieval results: which indicators are most retrieved by all retrievers (in Indicator Occurrences), and whether StockLLM-1B-Instruct makes predictions by analyzing candidates or just based on the candidates’ movements (in Appendix G.1). We also visualize indicator sequence embeddings in Appendix G.2. Moreover, we explore the performance of these retrieval methods with a larger size of StockLLM (in Appendix G.3).

In this section, we analyze the retrieved indicators of all RAG models. We calculate indicator occurrences on the ACL18 test set, and the results are shown in Figure 3. As shown in the figure, FinSeer is the only model that successfully extracts a diverse and comprehensive set of indicators while achieving superior performance. This clearly demonstrates its advanced temporal retrieval capabilities. Specifically, while other models like LLM Embedder and Instructor predominantly focus on basic indicators such as close price and adjusted close price, FinSeer effectively identifies and retrieves a wide range of technical indicators, including kdj crossover, MACD Histogram, Bollinger Bands, and various alpha factors. This richer set of retrieved indicators provides FinSeer with more comprehensive auxiliary information, enabling more accurate and reliable predictions.

Refer to caption
Figure 3:Indicator occurrences of different RAG models on ACL18 dataset.
4.4Case Study
This case study illustrates the critical importance of alignment between the retriever and the LLM’s forecasting preferences in financial time-series analysis. We examine the stock XOM on 2015-06-25 from the ACL18 dataset, where the adjusted close price exhibited a pronounced downward trend. The query sequence is as follows:

{ "query_stock": "XOM",
"query_date": "2015-06-25",
"recent_date_list": ["2015-06-18", "2015-06-19", "2015-06-22", "2015-06-23", "2015-06-24"],
"adjusted_close_list": [58.0813, 57.8979, 57.8707, 57.8027, 57.5377]}
While multiple retrievers were evaluated, only FinSeer successfully enabled StockLLM to predict the correct movement as a fall. Specifically, FinSeer retrieved five diverse indicators: close price, adjusted close price, alpha021, alpha054, and the highest price, providing a comprehensive view of the stock’s behavior. Alpha021 identifies trends based on short- and long-term price averages and volume conditions, while alpha054 combines price and volume rankings to assess performance within a specific time window. These indicators allowed StockLLM to accurately assess whether the downward trend would persist or reverse, demonstrating the value of retrieving contextually relevant and diverse features.

In contrast, other retrievers, such as Instructor, BGE, LLM Embedder, and E5, extracted sequences dominated by close or adjusted close prices, all reflecting similar downward trends. While these sequences aligned with the current trend, they failed to provide actionable insights for forecasting future movements, leading StockLLM to misinterpret them as noise and incorrectly predict a rise. Similarly, UAE retrieved sequences indicating overbought and oversold conditions, including three rise and two freeze trends. Although overbought signals often suggest a potential downturn, the retrieved sequences themselves exhibited rising or frozen trends, confusing StockLLM and resulting in an erroneous prediction. This case study underscores the superiority of FinSeer in retrieving meaningful and diverse indicators that align with the LLM’s forecasting logic, enabling more accurate and reliable predictions.

5Conclusion
In this paper, we present FinSrag, the first retrieval-augmented generation framework tailored for financial time-series forecasting. At its core, FinSeer, a novel retriever refined by LLM feedback, effectively identifies historically influential market sequences while filtering out financial noise. Combined with StockLLM, a fine-tuned LLM with 1B parameters, our framework leverages enriched financial datasets to capture previously overlooked market dynamics. Empirical results confirm that FinSeer surpasses both textual and distance-based retrievers in improving StockLLM’s prediction accuracy, highlighting the necessity of domain-specific retrieval in financial forecasting. Beyond stock markets, FinSrag establishes a blueprint for integrating RAG in time-sensitive decision-making domains.