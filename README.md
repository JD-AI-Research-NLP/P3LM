# P2DeNet
The code for the EMNLP2021 paper：[P^2DeNet: Permutation over Prophet Decoding Net for Generative Pre-training](XXX)
![image](https://user-images.githubusercontent.com/14817331/120296613-e1276500-c2fa-11eb-883d-b3a1f0db76c9.png)

## Requirements:
python==3.6.8
pytorch==1.5.0+cu101

## GLGE Data:
[download here](https://microsoft.github.io/glge/) 

## Pretrained Model and Reported Results:
[download here](xxx)

## Reproduce the Reported Results:
1. Pre-process the GLGE data and put it in the ./P2DeNet/glge/ folder.
2. Download the trained models, and place them in ./P2DeNet/glge/models/\[DATASET\]/ respectively.
3. Run the following command:
```
python finetune.generation.py test \[DATASET\]
```

## Train the Model:
```
python finetune.generation.py train \[DATASET\]
```

## Our Results
![image](![glge result](https://user-images.githubusercontent.com/14817331/120298865-05844100-c2fd-11eb-890d-a5410d846df8.png)
)

## Reference:
Thanks for your citation:
```
@inproceedings{wang-etal-2020-learning-decouple,
    title = "Learning to Decouple Relations: Few-Shot Relation Classification with Entity-Guided Attention and Confusion-Aware Training",
    author = "Bao, Junwei  and
      Wang, Yifan  and
      Ying, Jiangyong  and
      Gong, Yeyun  and
      Zhao, Jing  and
      Wu, Youzheng  and
      He, Xiaodong  and
      Zhou, Bowen",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2021",
    abstract = "Conventional left-to-right (L2R) generative pre-training methods face two issues during decoding: limited to unidirectional target sequence modeling, and constrained on strong local dependencies. In this paper, we propose P2DeNet, a permutation over prophet decoding net, which strengthens the modeling of bi-directional information and long token dependencies in target sequences, for generative pre-training. Specifically, P2DeNet learns to generate tokens in permuted order upon an order-aware transformer decoder, as well as the corresponding future N tokens with a multi-stream attention mechanism. Extensive experiments are conducted on the GLGE benchmark, which includes four datasets for summarization, two for question generation, one for conversational question answering, and one for dialog response generation, where P2DeNet achieves state-of-the-art results compared with published methods.",
}
```