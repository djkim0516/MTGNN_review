# MT-GNN(Multivariate Time Series Forecasting with Graph Neural Networks)

## Paper PDF file

[Connecting the Dots_MT-GNN.pdf](MTGNN_review/Connecting_the_Dots_MT-GNN.pdf)

[MT-GNN_review_김동준.pdf](MTGNN_review/MT-GNN_review_%EA%B9%80%EB%8F%99%EC%A4%80.pdf)

## Abstract

다변량 시계열 분석 - 다양한 분야 연구자들이 관심

기본적인 가정 : Its variables depend on one another. → 기존 기법로는 fully 반영 어려움

### GNN

- relational dependencies를 다루는 데에 큰 가능성을 보임
- Requires well-defined graph structures for info. prop.
- if dependencies not known in advance → cannot be applied

### 논문의 기법

- Auotmatically extracts the uni-directed relations among variables by graph learning module
- external knowledge can be easily integrated
- **Mix-hop propagation layer** - spatial dependencies
- **Dilated inception layer** - temporal dependencies

Graph learning, Graph convolution, temporal convolution modules are jointly learned.

## 1. Introduction

Multivariate time series forecasting methods → inherently assume interdependencies among variables

### Challenges

1. **No pre-defined graph structure**
    1. graph learning layer
    2. graph convolution module
    3. temporal convolution module
2. 대부분의 GNN은 **graph structure가 최적이지 않을 수 있다는 사실을 간과**함
    1. learn the internal graph structure
    2. learning algorithm using curriculum learning strategy : split multivariate time series into subgroups

Generally applicable to small/large graphs, short/long time series, with/without external graph

### Main Contributions

- graph learning module
- joint framework

## 2. Backgrounds

### 2.1 Multivariate Time Series Forecasting

**Statistical models**

- strong assumptions
- stationery process

Deep Learning more effective

CNN → can't exploit latent dependency btw pairs & variables

### 2.2 Graph Neural Networks

message passing, info prop, graph conv

## 3. Problem Formulation

${\bold z}_t∈{\bold R}^N$ : value of mulivariate var. of dim-N at time step t

inputs : $\chi = \{ \bold S_{t_1}, \bold S_{t_2}, \cdots , \bold S_{t_p} \}$

goal : $\Upsilon = \{ \bold z_{t_{P+1}}, \bold z_{t_{P+2}}, \cdots,\bold z_{t_{P+Q}}\}$

$\bold S_{t_i} \in \bold R^{N \times D}$

$D$ = feature dimension

Aim : build a mapping $f(\cdot)$ from $\chi$ to $Y$ minimizing the abs loss with l2 regularization

**def. 3.1 Graph :** $G=(V,E)$

V : set of nodes, E : set of edges, N : #nodes

**def. 3.2 Node Neighborhood** : $N(v)=\{u \in V|(v,u) \in E\}$

$v \in V$ : node 

$e=(v,u)\in E$ : edge pointing from **u** → to **v**

**def. 3.3 Adjacency Matrix : $\bold A \in R^{N \times N}$**

$A_{ij}=c \ > \ 0$ if ($(v_i, v_j) \in E$, else 0

## 4. Framework of MTGNN

### 4.1 Model Architecture

![Untitled](MTGNN_review/Untitled.png)

**1 graph learning layer →** computes a graph adjacency matrix

![Untitled](MTGNN_review/Untitled%201.png)

**m graph convolution modules**

**m temporal convolution modules**

→ interleaved to capture spatial, temporal dependencies respectively

To avoid gradient vanishing : **residual connections** added from the inputs, **Skip connections** added

**Output modules** : projects the hidden features to the desired output dimension

### 4.2 Graph Learning Layer

기존 방법의 문제점

1.  **high time & space complexity with $O(N^2)$**
    - **Sampling Approach :** only calculates pair-wise relationships
2. **distance metric often symmetric or bi-directional**
    - we suppose the relation to be **uni-directional**
    
    ![Untitled](MTGNN_review/Untitled%202.png)
    

(1, 2) $E_1, E_2$ : randomly initialized node embeddings, learnable

(5) $argtopk(\cdot)$ : returns the index of the top-k largest values of a vector

Asymmetric property achieved by Eq.(3) - Subtraction, ReLU

Eq. (5, 6) : makes the adjacency matrix sparse → non-connected nodes as zero

**Incorporate External Data**

$E_1, E_2 = Z$ 로도 설정 가능

몇몇 연구들은 dynamic spatial dependencies 고려 → 수렴 매우 어려움

우리 연구는 stable하고 interpretable한 node relationship 추출 가능

Our graph adjacency matrix is adaptable to change as new training data updates the model parameters.

### 4.3 Graph Convolution Module

Aims to fuse a node's info with its neighbors' info → to handle spatial dependencies

**Two mix-hop propagation layers**

![Untitled](MTGNN_review/Untitled%203.png)

Information propagation step : $\bold H^{(k)} \ =\ \beta \bold H_{in} +(1- \beta) \bold {\tilde A} \bold H^{(k-1)}   \ \ \ \ \ \ \ \ \ \ \ (7)$

Information selection step : $\bold H_{out} \ = \ \Sigma^K_{i=0} \bold H^{(k)} \bold W^{(k)} \ \ \ \ \ \ \ \ \ (8)$

$\bold H_{out}$ : output hidden states of the current layer

$\bold H^{(0)}  = \bold H_{in}$

$\bold {\tilde A} = \bold {\tilde D}^{-1}(\bold A + \bold I)$

$\bold {\tilde D}_{ii} = 1 + \Sigma_j \bold A_{ij}$

**First propagates info horizontally, selects info vertically.**

Eq.(7) 만 적용시 : some node info will be lost

spatial dependencies 정보 없다고 가정 시 : neighbor info 더하는 것은 불필요한 노이즈 증가

Eq.(8) : selection step needed

$\bold W^{(k)}$ : feature selector

만약 given graph structure가 spatial dependencies 없다면 : W^(k) = 0 

**Connection to existing works**

Kapoor : two mix-hop layers has the capability to represent the delta difference → concat

Ours : one mix-hop propagation layer has the same effect.

![Untitled](MTGNN_review/Untitled%204.png)

Summation is more efficient.

### 4.4 Temporal Convolution Module

set of standard dilated 1-D convolution filters

![Untitled](MTGNN_review/Untitled%205.png)

- Choosing the right kernel size is challenging.
- The receptive field size of a conv network grows linearly.

![Untitled](MTGNN_review/Untitled%206.png)

![Untitled](MTGNN_review/Untitled%207.png)

![Untitled](MTGNN_review/Untitled%208.png)

### 4.5 Skip Connection Layer & Output Module

$1 \times L_i$ standard convolutions → 1 or $Q$ 

### 4.6 Proposed Learning Algorithm

1. 중간 단계 변수값 다 저장 시 메모리 초과 가능성
    - randomly split the nodes into several groups
    - let the algorithm learn a sub-graph structure
    
    reduce the time, space complexity      $O(N^2)  \rightarrow O(N/s)^2$
    
2. 더 쉽게 local optimum에 도달할 수 있도록 하고픔
- curriculum learning
    - starts with solving the easiest problem → predict only the next step
    - increase the prediction length gradually

![Untitled](MTGNN_review/Untitled%209.png)

![Untitled](MTGNN_review/Untitled%2010.png)