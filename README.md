# WriteLikeYou

This repository contains the code of paper [Write Like You: Synthesizing Your Cursive Online Chinese Handwriting via Metric-based Meta Learning (Computer Graphics Forum, 2021)](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.142621).

## Abstract

In this paper, we propose a novel Sequence-to-Sequence model based on metric-based meta learning for the arbitrary style transfer of online Chinese handwritings. Unlike most existing methods that treat Chinese handwritings as images and are unable to reﬂect the human writing process, the proposed model directly handles sequential online Chinese handwritings. Generally, our model consists of three sub-models: a content encoder, a style encoder and a decoder, which are all Recurrent Neural Networks. In order to adaptively obtain the style information, we introduce an attention-based adaptive style block which has been experimentally proven to bring considerable improvement to our model. In addition, to disentangle the latent style information from characters written by any writers effectively, we adopt metric-based meta learning and pre-train the style encoder using a carefully-designed discriminative loss function. Then, our entire model is trained in an end-to-end manner and the decoder adaptively receives the style information from the style encoder and the content information from the content encoder to synthesize the target output. Finally, by feeding the trained model with a content character and several characters written by a given user, our model can write that Chinese character in the user’s handwriting style by drawing strokes one by one like humans. That is to say, as long as you write several Chinese character samples, our model can imitate your handwriting style when writing. In addition, after ﬁne-tuning the model with a few samples, it can generate more realistic handwritings that are difﬁcult to be distinguished from the real ones. Both qualitative and quantitative experiments demonstrate the effectiveness and superiority of our method.

## Requirements
```
python=3.6
tensorflow-gpu=1.11.0
```

## Citation
```
@inproceedings{tang2021write,
  title={Write Like You: Synthesizing Your Cursive Online Chinese Handwriting via Metric-based Meta Learning},
  author={Tang, Shusen and Lian, Zhouhui},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={2},
  pages={141--151},
  year={2021},
  organization={Wiley Online Library}
}
```
