## Louvain
![louvain](https://cdn.jsdelivr.net/gh/R1szW7xqoVdA/Vu-jewdLEw--f-_-/img/202211011142238.png)  
louvain算法是一种基于模块度的算法  
每一步的步骤如上图所示，
1. 对于每个节点，先计算这个节点i从原来的社区中脱离出来，然后加入与其直接相连的周围社区C，然后计算模块增益(the gain of modularity)  
$$
\Delta Q = [\frac{\sum_{in} + k_{i, in}}{2m} - (\frac{\sum_{tot} + k_i}{2m})^2] - [\frac{\sum_{in}}{2m} - (\frac{\sum_{tot}}{2m})^2 - (\frac{k_i}{2m})^2]
$$
其中$\sum_{in}$表示C内部所有连边的权重之和，$k_{i, in}$表示节点i与社区C内节点的连边的权重之和  
$\sum_{tot}$表示社区C中所有节点和外界的连边的权重之和，$k_i$表示与节点i连接的所有边的权重之和  
$m$表示所有边的权重之和  
加入增益最大的社区(如果计算得到的所有增益全都是负值，则不加入任何社区)  
2. 在计算完所有节点的增益并分配好社区后，将一个社区汇聚成一个节点，用于进行下一次的计算  

不断计算直到达到设定的分辨率阈值(模块度)  
$$
Q = \frac{1}{2m} \sum_{i, j}[A_{ij} - \frac{k_ik_j}{2m}]\delta(c_i, c_j)
$$
其中$A_{ij}$表示节点i和j之间的权重，$\delta(c_i, c_j)$表示如果节点i和j属于同一个社区，则为1，否则为0

### 在ScanPy中使用的流程
在Scanpy中，louvain只是作为图聚类的方法，由于这个方法及其依赖于图的结构，所以需要重点关注的是```sc.pp.neighbors(adata)```这个生成图的方法  
Scanpy中使用的方法是对特征矩阵中的每两行两两求乘积作为边的权重，然后选取k(这里的k默认为30)个权重最大的边作为最终生成的边

## STAGATE
![STAGATE](https://cdn.jsdelivr.net/gh/R1szW7xqoVdA/Vu-jewdLEw--f-_-/img/202211011143944.png)
使用类似于自编码器的结构，在聚合邻边信息时引入attention  

- 生成图:  
使用KNN或是给定半径  
在生成图时可以先进行一次pre-cluster，切断聚类中不同类之间的边  

- GNN:  
如结构图中所示，使用一个类似自编码器的结构  
在编码的过程中:  
对于每次聚合 $h^{(k)}_i = \sigma(\sum_{j\in S_i} att^{(k)}_{ij} (W_kh^{(k-1)}_j))$  
最后一层全连接 $h^{(k)}_i = \sigma(W_kh^{(k-1)}_j)$  
解码过程和编码过程类似，不同的是解码时使用的att值和编码时相同，解码时使用的全连接层是编码时使用的全连接层的转置   
attention机制:  
训练邻边和自身的向量$v_s$和$v_r$  
$$
e^{(k)}_{ij} = \sigma(v^{(k)^T}_s (W_kh^{k-1}_i) + v^{(k)^T}_r (W_kh^{k-1}_j)), \\
att^{(k)}_{ij} = \frac{exp(e^{(k)}_{ij})}{\sum_{i \in N(i)}e^{(k)}_{ij}},
$$
如果使用到了Construction of cell type-aware SNN，则只需要按设定好的$\alpha$将两个attention相加即可
$$
att_{ij} = (1 - \alpha)att^{spatial}_{ij} + \alpha att^{aware}_{ij}
$$

- loss:  
主要思想是经过编码/解码后得到的值应该和原值类似，所以loss函数设置为前后两向量的距离$\sum^N_{i=1} || x_i - \hat{h}^0_i ||_2$

## CCST
![CCST](https://cdn.jsdelivr.net/gh/R1szW7xqoVdA/Vu-jewdLEw--f-_-/img/202211011144775.png)
使用DGI模型对空间转录组数据进行处理    

- 生成图:  
给定超参数$d_{thres}$，计算每两个节点间的距离，如果小于超参数，则生成边，否则不生成边。  
为了平衡连接边的权重和节点的基因表达信息，引入超参数$\lambda$，修正邻接矩阵$A = \lambda * I + (1 - \lambda) * A_0$  

- 数据预处理:   
对于每个数据集，需要删除表达量较低的基因，对每个spot的基因表达进行归一化。

- GNN:  
采用DGI模型，loss函数：
$$Loss = \sum^N_{i = 1} E_{X, A}[\log D(h_i, s)] + E_{X, \overline{A}}[\log D(\overline{h_i}, s)]$$ 
其中:  
X为每个细胞的基因表达矩阵。  
A为正确图的邻接矩阵。  
$\overline{A}$为生成的混淆图的邻接矩阵(这里混淆图的边是随机生成的)。  
$h_i$是正常图经过GCN后得到的embedding vector。  
$\overline{h_i}$是混淆图经过同一个GCN后得到的embedding vector。  
s是正常图经过GCN后得到所有embedding vector的平均值(用来代表整张图)。  
DGI的主要思想是*最大化* **混淆图生成的向量**和图向量间的距离，同时*最小化* **正常图生成的向量**和图向量之间的距离。  

- 聚类:  
得到embedding vector后先通过PCA进行降维，然后使用umap进行聚类和可视化。  

- Differential gene expression analysis:  
秩和检验

## SpaGCN
![](https://cdn.jsdelivr.net/gh/R1szW7xqoVdA/Vu-jewdLEw--f-_-/img/202211011145644.jpg)

- 生成图:  
通过计算节点间的欧几里得距离来生成一个带权重的图，其中u, v之间的权重计算公式为:  
$$
w(u, v) = exp (-\frac{d(u, v)^2}{2l^2})
$$
这里的$l$是一个用来控制图权重的超参数，通过调整$l$让每个节点的连接边的权重和相似。  
在图生成时，可以加入组织学(histology)信息，将每个节点构成一个三维的节点：  
对于每个节点，考虑组织学图片中，**以这个节点为中心**的50*50内所有像素点对rgb信息，计算第三个维度z的公式如下  
$$
z_v = \frac{mean(r) * var(r) + mean(g) * var(g) + mean(b) * var(b)}{var(r) + var(g) + var(b)}\\
$$
rescale:
$$
z^*_v = \frac{z_v + mean(z)}{std(z)} * \max(std(x), std(y)) * s
$$
其中mean表示均值，var表示方差，std表示标准差，这里的s是一个用于平衡z与x, y大小关系的超参数。  
得到第三个维度z后，计算欧几里得距离，并使用最上面的公式计算边权重。

- GCN:  
在进行节点信息聚合之前，先对网络进行一次聚类(Louvain或KMeans)并得到每个类的中心特征(类中所有节点特征的平均值)  
使用传统的GCN聚合邻居边的信息  
loss:  
首先计算每个节点和每个聚类中心的距离(可以认为是节点i是聚类j的概率):  
$$
q_{ij} = \frac{(1 + h_i - \mu^2_j)^{-1}}{\sum^K_{j'=1}(1 + h_i - \mu^2_{j'})^{-1}}
$$
设置辅助目标分布函数:  
$$
p_{ij} = \frac{q_{ij} / \sum^N_{i = 1} q_{ij}}{\sum^K_{j' = 1} (q_{ij'} / \sum^N_{i = 1} q_{ij'})}
$$
通过最小化这两个函数的KL散度来更新GCN中的参数以及聚类中心。  
$$
L = KL(P||Q) = \sum^N_i \sum^K_j p_{ij}\log\frac{p_{ij}}{q_{ij}}
$$

- 查找SVG:  
秩和检验
