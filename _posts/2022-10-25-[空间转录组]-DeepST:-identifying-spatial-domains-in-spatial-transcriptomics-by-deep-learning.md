## DeepST流程图

![DeepST流程图](https://cdn.jsdelivr.net/gh/R1szW7xqoVdA/Vu-jewdLEw--f-_-/img/202210241628242.png)

## 数据预处理

DeepST用到了基因表达数据、细胞位置信息数据以及morphology图像信息，下面先介绍DeepST如何处理图像信息

### DeepST处理图像信息

- 注: 10x Visium数据的存储结构
  `adata.obsm['spatial']`中保存的是每个spot在morphology图像中的位置信息。
  `adata.obs['array_row']`以及 `adata.obs['array_col']`中保存的是这个spot在图中array的相应位置。
  `adata.uns["spatial"][library_id]["scalefactors"]`中保存了各种分辨率下图片与位置坐标的scalefactors。

获取每个spot周围50个pixel的rgb信息的核心代码如下

```python
for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
    # 对于每个spot，查找其周围50*50位置的坐标点
    imagerow_down = imagerow - crop_size / 2
    imagerow_up = imagerow + crop_size / 2
    imagecol_left = imagecol - crop_size / 2
    imagecol_right = imagecol + crop_size / 2

    # 将这部分图片剪裁下载
    tile = img_pillow.crop(
        (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
    tile.thumbnail((target_size, target_size), Image.ANTIALIAS) ##### 
    tile.resize((target_size, target_size)) ###### 

    # 保存图片以及图片的地址
    tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
    out_tile = Path(save_path) / (tile_name + ".png")
    tile_names.append(str(out_tile))
    if verbose:
        print(
            "generate tile at location ({}, {})".format(
                str(imagecol), str(imagerow)))
    tile.save(out_tile, "PNG")
```

- 代码实现:  
`DeepST.run._get_adata()` 
其中`read_10X_Visium()`用于读取10x Visium数据    
`image_crop()`用于剪裁和保存每个spot的morphology图像信息  
`image_feature.extract_image_feat()`用于使用预训练的CNN模型计算morphology图像的embedding信息  

## 图生成

### 基因表达权重

计算基因i与基因j之间的表达的权重参数$GC_{ij}$:

$$
GC_{ij} = 1 - \frac{(S_i - \overline{S_i})*(S_j - \overline{S_j})}{||(S_i - \overline{S_i})||_2*||(S_j - \overline{S_j})||_2}
$$

-> 这里可以看作是1 - 数据标准化后的基因表达向量的两两cos值

### morphology图像信息权重

计算基因i与基因j在morphology图像中的权重，步骤如下:

1. 截取得到每个spot周围50*50的morphology图像信息
2. 将截取到的图像通过一个CNN预训练模型，得到embedding
3. 使用PCA降维
4. 计算每两个spot之间的权重$MS_{ij}$:

$$
MS_{ij} = 1 - \frac{S_i*S_j}{||S_i||_2*||S_j||_2}
$$

### 空间信息权重（邻接矩阵）

计算每两个spot之间的空间距离，然后按照设定的阈值得到邻接矩阵

- 以上计算三种权重的代码实现:  
`augment.cal_weight_matrix()`

### 最终的经过增强的基因表达

$$
\tilde{GE_i} = GE_i + \frac{\sum^n_{j=1}GE_j*GC_{ij}*MS_{ij}*adj_{ij}}{n}
$$

- 代码实现:  
`augment.find_adjacent_spot()`
`augment.augment_gene_data()`

- 注: 这里的这个n应该是指定的邻接节点的个数的n，不是全部的n个spot。


## 编码器部分
在进入编码器之前，需要先对基因表达矩阵进行一系列预处理，然后还需要对矩阵进行去噪（去噪的方法是随机给矩阵的一些位置置为0）。

### 基因表达的编码器
- 编码器:  
$$
E(X) = Z_g, X \in R^{N*M}, Z_g \in R^{N×R}
$$

- 解码器:  
$$
Z_g'=Z_g+Z, \\
D(Z_g')=X', Z_g′ \in R^{N*(R+R)}, X' \in R^{N*M}
$$
这里的$Z$会在后面说明，是GCN部分编码器得到的embedding  

每层encoder/decoder中包含`nn.Linear()`, `nn.BatchNorm1d()`, 一个激活函数(ELU或sigmod)以及`nn.Dropout()`

- Loss:  
最小化原空间表达和经过编解码后得到的结果的差距  
$$
L = \frac{1}{N} \sum^N_{i=1}|| X_i - D(E(X_i)) ||^2
$$

### 基于GCN的编码器
- 编码器:  
和普通的GCN一样，需要基因表达矩阵以及邻接矩阵  
$$
\tilde{X} = GNN(X,A) = \tilde{A} ReLU(\tilde{A}XW_0) W_1, \\
\tilde{A} = D^{−\frac{1}{2}}AD^{−\frac{1}{2}},
$$
然后计算两个参数$\mu$和$\log{\sigma^2}$:  
$$
\mu = GNN_{\mu}(X, A) = \tilde{A}\overline{X}W_2, \\
\log{\sigma^2} = GNN_{\sigma}(X, A) = \tilde{A}\overline{X}W_2
$$
在源码中分别为`DeepST_model.conv()`, `DeepST_model.conv_mean()`以及`DeepST_model.conv_logvar()`  
- 注: 这里的$W_2$虽然使用的是相同的符号，但是使用的是不同的两个矩阵  

- $Z$:
刚才所说的$Z$由参数$\mu$和$\log{\sigma^2}$生成:  
$$
Z = \mu + \log{\sigma^2} * \epsilon
$$
其中$\epsilon$~$N(0, 1)$

- 解码器:  
按照节点间特征的相似度判断是否生成边:  
$$
p(A|Z) = \prod^N_{i=1} \prod^N_{j=1}p(a_{ij} | z_i, z_j), \\
p(a_{ij} | z_i, z_j) = \sigma{(ZZ^T)}
$$
源码实现: `InnerProductDecoder.forward()`

- Loss:  
$$
L_g =E_{q(Z|X, A)}[\log{p(A|Z)}]−KL[q(Z|X, A)||p(Z)]
$$
这里使用的是VAE的方式训练这部分的损失  

- 注: 两个自编码器来自同一个模型，所以两个损失函数的值按照一定比例求和后再进行反向传播。  


## domain classifier  
用于识别输入spot的domain，这里使用了DAN的相关方法，每层使用的是GRL(为了反向传播时的梯度是负的)  
$$
G_d(x; W, b) = sigm(Wx + b), \\
L_d = -\frac{1}{N}\sum_i\sum^M_{d=1}D_{id}\log{p_{id}}
$$
这里使用到的损失函数是交叉熵损失函数


## early stop
这里DeepST使用DEC（Deep Embedded Clustering）进行预聚类，如果聚类结果和上次的结果足够相似，则early stop。  
- 这里进行DEC的步骤是先计算q(在`DeepST_model.forward()`中)，然后计算目标分布p(`DeepST_model.target_distribution()`)，将p作为最终得到的label与上一轮的到的label（如果还没计算过p，则用在训练前先使用kmeans或louvain的聚类结果）进行比较，小于指定的超参数则停止训练。  

## 自编码器Loss函数总结
这里的自编码器用到了三种loss函数，分别是AE得到的loss、VAE得到的loss以及最后DEC得到的q和p计算的KL散度。最终将三者相加即为最后得到的loss。  
- 注: 如果需要进行domain classify，则还需要再加上这部分的loss（CE）。
