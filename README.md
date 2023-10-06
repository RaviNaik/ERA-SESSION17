# ERA-SESSION17 BERT, GPT & ViT using a single Transformer Architecture in Pytorch-Lightning

### Tasks:
1. :heavy_check_mark: Re-write the code in such a way where there is one transformer.py file that you can use to train all 3 model.
2.  :heavy_check_mark: Share the last few log details for all 3 models (BERT, GPT and ViT)
3.   :heavy_check_mark: Share the link to a single repo where all 3 ipynb files can be found

### BERT
**Architecture:**
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/7bdad8c3-ba7e-497a-b735-8f2ce7638701)

#### Model Summary:
```python
   | Name                             | Type                    | Params
------------------------------------------------------------------------------
0  | embedding                        | Embeddings              | 5.1 M 
1  | embedding.lut                    | Embedding               | 5.1 M 
2  | pe                               | PositionalEncoding      | 0     
3  | embedding_dropout                | Dropout                 | 0     
4  | transformer_layers               | Sequential              | 1.6 M 
5  | transformer_layers.0             | TransformerLayer        | 198 K 
6  | transformer_layers.0.mha         | MultiheadAttention      | 66.0 K
7  | transformer_layers.0.ln1         | LayerNorm               | 256   
8  | transformer_layers.0.ln2         | LayerNorm               | 256   
9  | transformer_layers.0.feedforward | PositionwiseFeedForward | 131 K 
10 | transformer_layers.1             | TransformerLayer        | 198 K 
11 | transformer_layers.1.mha         | MultiheadAttention      | 66.0 K
12 | transformer_layers.1.ln1         | LayerNorm               | 256   
13 | transformer_layers.1.ln2         | LayerNorm               | 256   
14 | transformer_layers.1.feedforward | PositionwiseFeedForward | 131 K 
15 | transformer_layers.2             | TransformerLayer        | 198 K 
16 | transformer_layers.2.mha         | MultiheadAttention      | 66.0 K
17 | transformer_layers.2.ln1         | LayerNorm               | 256   
18 | transformer_layers.2.ln2         | LayerNorm               | 256   
19 | transformer_layers.2.feedforward | PositionwiseFeedForward | 131 K 
20 | transformer_layers.3             | TransformerLayer        | 198 K 
21 | transformer_layers.3.mha         | MultiheadAttention      | 66.0 K
22 | transformer_layers.3.ln1         | LayerNorm               | 256   
23 | transformer_layers.3.ln2         | LayerNorm               | 256   
24 | transformer_layers.3.feedforward | PositionwiseFeedForward | 131 K 
25 | transformer_layers.4             | TransformerLayer        | 198 K 
26 | transformer_layers.4.mha         | MultiheadAttention      | 66.0 K
27 | transformer_layers.4.ln1         | LayerNorm               | 256   
28 | transformer_layers.4.ln2         | LayerNorm               | 256   
29 | transformer_layers.4.feedforward | PositionwiseFeedForward | 131 K 
30 | transformer_layers.5             | TransformerLayer        | 198 K 
31 | transformer_layers.5.mha         | MultiheadAttention      | 66.0 K
32 | transformer_layers.5.ln1         | LayerNorm               | 256   
33 | transformer_layers.5.ln2         | LayerNorm               | 256   
34 | transformer_layers.5.feedforward | PositionwiseFeedForward | 131 K 
35 | transformer_layers.6             | TransformerLayer        | 198 K 
36 | transformer_layers.6.mha         | MultiheadAttention      | 66.0 K
37 | transformer_layers.6.ln1         | LayerNorm               | 256   
38 | transformer_layers.6.ln2         | LayerNorm               | 256   
39 | transformer_layers.6.feedforward | PositionwiseFeedForward | 131 K 
40 | transformer_layers.7             | TransformerLayer        | 198 K 
41 | transformer_layers.7.mha         | MultiheadAttention      | 66.0 K
42 | transformer_layers.7.ln1         | LayerNorm               | 256   
43 | transformer_layers.7.ln2         | LayerNorm               | 256   
44 | transformer_layers.7.feedforward | PositionwiseFeedForward | 131 K 
45 | lm_head                          | Linear                  | 5.2 M 
46 | softmax                          | Softmax                 | 0     
------------------------------------------------------------------------------
11.9 M    Trainable params
0         Non-trainable params
11.9 M    Total params
47.465    Total estimated model params size (MB)
```
**Training Logs**:
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/e9b85c19-389f-4e02-affc-e722000dcf13)
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/3311ba90-fddf-4694-9084-ea07d7488e4a)


### GPT
**Architecture**:
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/cff53789-5937-4dee-9536-92d3adf30213)


#### Model Summary:
```python
   | Name                             | Type                    | Params
------------------------------------------------------------------------------
0  | embedding                        | Embeddings              | 23.4 M
1  | embedding.lut                    | Embedding               | 23.4 M
2  | pe                               | PositionalEncoding      | 0     
3  | embedding_dropout                | Dropout                 | 0     
4  | transformer_layers               | Sequential              | 42.5 M
5  | transformer_layers.0             | TransformerLayer        | 7.1 M 
6  | transformer_layers.0.mha         | MultiheadAttention      | 2.4 M 
7  | transformer_layers.0.ln1         | LayerNorm               | 1.5 K 
8  | transformer_layers.0.ln2         | LayerNorm               | 1.5 K 
9  | transformer_layers.0.feedforward | PositionwiseFeedForward | 4.7 M 
10 | transformer_layers.1             | TransformerLayer        | 7.1 M 
11 | transformer_layers.1.mha         | MultiheadAttention      | 2.4 M 
12 | transformer_layers.1.ln1         | LayerNorm               | 1.5 K 
13 | transformer_layers.1.ln2         | LayerNorm               | 1.5 K 
14 | transformer_layers.1.feedforward | PositionwiseFeedForward | 4.7 M 
15 | transformer_layers.2             | TransformerLayer        | 7.1 M 
16 | transformer_layers.2.mha         | MultiheadAttention      | 2.4 M 
17 | transformer_layers.2.ln1         | LayerNorm               | 1.5 K 
18 | transformer_layers.2.ln2         | LayerNorm               | 1.5 K 
19 | transformer_layers.2.feedforward | PositionwiseFeedForward | 4.7 M 
20 | transformer_layers.3             | TransformerLayer        | 7.1 M 
21 | transformer_layers.3.mha         | MultiheadAttention      | 2.4 M 
22 | transformer_layers.3.ln1         | LayerNorm               | 1.5 K 
23 | transformer_layers.3.ln2         | LayerNorm               | 1.5 K 
24 | transformer_layers.3.feedforward | PositionwiseFeedForward | 4.7 M 
25 | transformer_layers.4             | TransformerLayer        | 7.1 M 
26 | transformer_layers.4.mha         | MultiheadAttention      | 2.4 M 
27 | transformer_layers.4.ln1         | LayerNorm               | 1.5 K 
28 | transformer_layers.4.ln2         | LayerNorm               | 1.5 K 
29 | transformer_layers.4.feedforward | PositionwiseFeedForward | 4.7 M 
30 | transformer_layers.5             | TransformerLayer        | 7.1 M 
31 | transformer_layers.5.mha         | MultiheadAttention      | 2.4 M 
32 | transformer_layers.5.ln1         | LayerNorm               | 1.5 K 
33 | transformer_layers.5.ln2         | LayerNorm               | 1.5 K 
34 | transformer_layers.5.feedforward | PositionwiseFeedForward | 4.7 M 
35 | lm_head                          | Linear                  | 23.5 M
36 | softmax                          | Softmax                 | 0     
------------------------------------------------------------------------------
89.4 M    Trainable params
0         Non-trainable params
89.4 M    Total params
357.758   Total estimated model params size (MB)
```
**Training Logs**:
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/e03b464d-2a96-493e-ad52-776560105aa4)
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/ede0b3c7-b71d-4221-aba8-d9899fb4ec19)

### ViT
**Architecture:**
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/70a22928-91a7-4b4c-b230-268621e2bfe8)

#### Model Summary:
```python
   | Name                             | Type                    | Params
------------------------------------------------------------------------------
0  | patch_embedding                  | PatchEncoding           | 0     
1  | patch_embedding.flatten          | Flatten                 | 0     
2  | linear_mapper                    | Linear                  | 590 K 
3  | pe                               | PositionalEncoding      | 0     
4  | embedding_dropout                | Dropout                 | 0     
5  | transformer_layers               | Sequential              | 42.5 M
6  | transformer_layers.0             | TransformerLayer        | 7.1 M 
7  | transformer_layers.0.mha         | MultiheadAttention      | 2.4 M 
8  | transformer_layers.0.ln1         | LayerNorm               | 1.5 K 
9  | transformer_layers.0.ln2         | LayerNorm               | 1.5 K 
10 | transformer_layers.0.feedforward | PositionwiseFeedForward | 4.7 M 
11 | transformer_layers.1             | TransformerLayer        | 7.1 M 
12 | transformer_layers.1.mha         | MultiheadAttention      | 2.4 M 
13 | transformer_layers.1.ln1         | LayerNorm               | 1.5 K 
14 | transformer_layers.1.ln2         | LayerNorm               | 1.5 K 
15 | transformer_layers.1.feedforward | PositionwiseFeedForward | 4.7 M 
16 | transformer_layers.2             | TransformerLayer        | 7.1 M 
17 | transformer_layers.2.mha         | MultiheadAttention      | 2.4 M 
18 | transformer_layers.2.ln1         | LayerNorm               | 1.5 K 
19 | transformer_layers.2.ln2         | LayerNorm               | 1.5 K 
20 | transformer_layers.2.feedforward | PositionwiseFeedForward | 4.7 M 
21 | transformer_layers.3             | TransformerLayer        | 7.1 M 
22 | transformer_layers.3.mha         | MultiheadAttention      | 2.4 M 
23 | transformer_layers.3.ln1         | LayerNorm               | 1.5 K 
24 | transformer_layers.3.ln2         | LayerNorm               | 1.5 K 
25 | transformer_layers.3.feedforward | PositionwiseFeedForward | 4.7 M 
26 | transformer_layers.4             | TransformerLayer        | 7.1 M 
27 | transformer_layers.4.mha         | MultiheadAttention      | 2.4 M 
28 | transformer_layers.4.ln1         | LayerNorm               | 1.5 K 
29 | transformer_layers.4.ln2         | LayerNorm               | 1.5 K 
30 | transformer_layers.4.feedforward | PositionwiseFeedForward | 4.7 M 
31 | transformer_layers.5             | TransformerLayer        | 7.1 M 
32 | transformer_layers.5.mha         | MultiheadAttention      | 2.4 M 
33 | transformer_layers.5.ln1         | LayerNorm               | 1.5 K 
34 | transformer_layers.5.ln2         | LayerNorm               | 1.5 K 
35 | transformer_layers.5.feedforward | PositionwiseFeedForward | 4.7 M 
36 | lm_head                          | Linear                  | 2.3 K 
37 | softmax                          | Softmax                 | 0     
------------------------------------------------------------------------------
43.1 M    Trainable params
0         Non-trainable params
43.1 M    Total params
172.484   Total estimated model params size (MB)
```
**Sample Data:**
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/645f3a6c-fcf9-4e67-906e-4e294839eca0)

**Patchified Image:**
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/c9ad1d32-8dde-4543-a075-39e66d5cf57a)

**Training Logs**:
![image](https://github.com/RaviNaik/ERA-SESSION17/assets/23289802/748d2640-171d-46da-b7d1-7b17d4e6c67f)

