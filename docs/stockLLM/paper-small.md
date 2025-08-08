# FinSrag: Retrieval-Augmented Financial Time Series Forecasting

## Overview

FinSrag is the first retrieval-augmented generation (RAG) framework specifically designed for financial time-series forecasting, featuring a novel domain-specific retriever called **FinSeer**. The framework addresses the critical challenge of retrieving relevant historical patterns from noisy financial data to enhance stock movement prediction accuracy.

## Core Problem

Traditional approaches to financial time-series forecasting face several limitations:

- **Superficial Pattern Matching**: Conventional embedding-based methods struggle with numeric sequences that appear similar but lack semantic meaning
- **Limited Context**: Distance-based methods like Dynamic Time Warping (DTW) rely on single-variable comparisons, ignoring rich contextual information
- **Noise Susceptibility**: Financial markets contain substantial noise that misleads traditional retrievers

## Architecture

### FinSeer Retriever

FinSeer is trained using LLM feedback to identify sequences containing essential or complementary information beneficial for prediction, even with limited surface similarity to the query.

#### Key Components:
1. **Candidate Selection Mechanism**: Uses LLM-guided relevance estimation
2. **Similarity-driven Training**: Aligns queries with historically influential sequences
3. **Financial Noise Filtering**: Filters out spurious correlations

### StockLLM

A 1B-parameter language model fine-tuned specifically for stock movement prediction, serving as the generative backbone.

## Mathematical Framework

### Stock Movement Classification

The framework predicts binary stock movements (rise/fall) using percentage returns:

$$R_t = \frac{\text{adj\_close}_d - \text{adj\_close}_{d-1}}{\text{adj\_close}_{d-1}} \times 100$$

Movement classification:

$M_{q,d} = \begin{cases} 
\text{rise}, & R_t > 0.55 \\
\text{fall}, & R_t < -0.5 \\
\text{freeze}, & -0.5 \leq R_t \leq 0.55
\end{cases}$

### Candidate Scoring and Selection

For each query $q$ and candidate $c_i$, the framework uses StockLLM to generate logits, converted to probabilities:

$$P(c) = \frac{e^{z_c}}{\sum_j e^{z_j}}$$

where $z_c$ is the logit for the correct class and $z_j$ represents logits for all possible classes.

### Training Objective

The retriever $R(q)$ is optimized to distinguish between positive sequences $\mathbb{C}^\mathbb{P}$ (top-1 scored) and negative sequences $\mathbb{C}^\mathbb{N}$ (bottom-15 scored):

$$R(q) = \arg\max_{s \in \mathbb{C}^\mathbb{P} \cup \mathbb{C}^\mathbb{N}} \text{sup}(q,s)$$

### Knowledge Distillation

The framework employs KL divergence to transfer knowledge from StockLLM to the retriever:

Normalized rewards using softmax with temperature $\alpha$:

$w_i = \text{softmax}_R\left(\frac{P(c_i)}{\alpha}\right)$

KL divergence optimization:

$\min \sum_c -w_i \times \log\left(\frac{\exp(\langle \mathbf{e}_q, \mathbf{e}_{c_i} \rangle / \tau)}{\sum_{c' \in \mathbb{C}} \exp(\langle \mathbf{e}_q, \mathbf{e}_{c'} \rangle / \tau)}\right)$

where $\mathbf{e}_q$ and $\mathbf{e}_{c_i}$ are query and candidate embeddings, and $\tau$ is the temperature parameter.

## Implementation Details

### Dataset Construction

The framework uses three hierarchical data layers:

1. **Core Price Metrics** (6 indicators): OHLCV data and adjusted close prices
2. **Primary Technical Indicators** (10 indicators): Log returns, momentum, VWAP
3. **Alpha Factors** (18 indicators): Selected via Mutual Information-based process

Total: **34 financial indicators** across three datasets (ACL18, BIGDATA22, STOCK23)

### Query and Candidate Construction

- **Temporal Range**: Each query requires at least one year of historical candidates
- **Sliding Window**: One-day sliding window for consecutive generation
- **Query Content**: Stock ID, query date, and 5-day adjusted closing prices
- **Candidate Content**: Stock ticker, price movement, financial indicator, and 5-day indicator values

### Sequence Serialization

Data is serialized in JSON format for LLM interpretation:

```json
{
  "query_stock": "MO",
  "query_date": "2015-06-02", 
  "recent_date_list": ["2015-05-26", "2015-05-27", "2015-05-28", "2015-05-29", "2015-06-01"],
  "adjusted_close_list": [29.669, 29.9872, 29.8657, 29.6227, 29.6227]
}
```

### Training Configuration

- **Base Model**: LLaMA 3.2-1B-Instruct fine-tuned with LoRA
- **Learning Rate**: 5e-5 with cosine scheduler
- **Training**: 5 epochs with mixed-precision (fp16)
- **Framework**: LlamaFactory for implementation

## Key Advantages

### 1. Domain-Specific Design
- First RAG framework tailored for financial time-series
- Incorporates 28 expert-selected technical indicators beyond price data
- Captures diverse market dynamics previously overlooked

### 2. Superior Performance
- Consistently outperforms 5 state-of-the-art retrievers from MTEB leaderboard
- Only method that consistently enhances LLM forecasting accuracy
- Demonstrates robustness across different market conditions (2014-2023)

### 3. Intelligent Pattern Recognition
- Identifies sequences with complementary predictive signals
- Moves beyond surface-level pattern matching
- Filters out financial noise effectively

### 4. Financial Insights
- Provides empirically grounded alternative to theoretical economic models
- Enables systematic identification of predictive market features
- Offers data-driven approach to study market inefficiencies

## Experimental Results

### Performance Comparison

| Method | ACL18 (ACC/MCC) | BIGDATA22 (ACC/MCC) | STOCK23 (ACC/MCC) |
|--------|-----------------|--------------------|--------------------|
| w/o Retrieval | 0.498/-0.006 | 0.493/-0.017 | 0.509/0.021 |
| DTW | 0.516/0.041 | 0.500/0.010 | 0.492/-0.007 |
| Instructor | 0.498/-0.005 | 0.493/-0.015 | 0.505/0.010 |
| E5-mistral-7b | 0.510/0.027 | 0.498/-0.002 | 0.499/-0.001 |
| **FinSeer** | **0.517/0.035** | **0.510/0.023** | **0.542/0.085** |

### Key Findings

1. **Consistent Enhancement**: FinSeer is the only method that consistently improves performance across all datasets
2. **Indicator Diversity**: Successfully extracts comprehensive set of indicators (KDJ crossover, MACD Histogram, Bollinger Bands, Alpha factors)
3. **Market Volatility Resilience**: Maintains performance during volatile periods (2022-2023) where DTW fails

## Limitations

### 1. Domain Specificity
- Current evaluation focuses on stock movement prediction
- Generalizability to other time-series domains requires validation
- Limited to financial market applications

### 2. Data Modality
- Focuses on unimodal time-series data
- Does not integrate textual reports, visual data, or other modalities
- Potential for multimodal enhancement unexplored

### 3. Market Coverage
- Evaluated primarily on high-volume US stocks
- Limited geographic and market segment coverage
- Scalability to global markets uncertain

### 4. Temporal Constraints
- Requires substantial historical data (minimum 1 year)
- Performance in data-sparse scenarios unknown
- Cold-start problem for new financial instruments

## Technical Implementation Notes

### Code Availability
- Framework available at: FinSrag Repository (code mentioned as available in paper)
- Licensed under MIT License
- Built using Python, PyTorch, and LlamaFactory

### Computational Requirements
- Training: NVIDIA A800 GPU with 80GB memory
- Inference: Optimized for real-time financial decision-making
- Scalable architecture for production deployment

### Evaluation Metrics
- **Accuracy (ACC)**: Overall prediction correctness
- **Matthews Correlation Coefficient (MCC)**: Balanced measure considering class distribution

## Future Directions

### 1. Cross-Domain Applications
- Extension to economic indicators, energy demand forecasting
- Epidemiological trend prediction
- General time-series forecasting tasks

### 2. Multimodal Integration
- Incorporation of news sentiment, social media data
- Visual chart pattern recognition
- Cross-modal information fusion

### 3. Advanced Retrieval Mechanisms
- Self-supervised learning approaches
- Dynamic candidate pool management
- Real-time adaptation to market regime changes

### 4. Scalability Enhancements
- Distributed training and inference
- Edge computing optimization
- Integration with existing trading systems

## References

Key citations from the research:

- Fama, E. F., & French, K. R. (2000). Forecasting profitability and earnings. *The Journal of Business*, 73(2), 161-175.
- Xie, Q., et al. (2023). The wall street neophyte: A zero-shot analysis of chatgpt over multimodal stock movement prediction challenges. *arXiv preprint arXiv:2304.05351*.
- Zhang, P., et al. (2023). Retrieve anything to augment large language models. *arXiv preprint arXiv:2310.07554*.
- Xu, Y., & Cohen, S. B. (2018). Stock movement prediction from tweets and historical prices. *ACL 2018*.
- Kakushadze, Z. (2016). 101 formulaic alphas. *Wilmott*, 2016(84), 72-81.

## Conclusion

FinSrag represents a significant advancement in financial time-series forecasting by introducing the first domain-specific RAG framework. The combination of FinSeer's intelligent retrieval capabilities and StockLLM's prediction power demonstrates superior performance across multiple datasets and market conditions. While limitations exist in domain coverage and data modality, the framework establishes a strong foundation for future research in time-sensitive decision-making domains and provides practical value for financial market applications.