赛题名称：CCF贝壳房产聊天问答匹配比赛

赛题链接：https://www.datafountain.cn/competitions/474

赛题类型：自然语言处理、文本分类

分享内容：比赛baseline

这里仅作为个人学习的baseline，整体思路如下：

尝试不同预训练模型(单模)作文本匹配，其中包括
[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)、
[BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)、
[RoBERTa-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)、
[RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm)、
[XLNet-base](https://github.com/ymcui/Chinese-XLNet)、
[XLNet-mid](https://github.com/ymcui/Chinese-XLNet)、
[NEZHA](https://github.com/lonePatient/NeZha_Chinese_PyTorch)、
[RoBERTa-zh-Large](https://github.com/brightmart/roberta_zh)
(**前六个模型均来自科大讯飞**)

单个模型采用5折交叉验证效果更佳。其中BERT-wwm-ext效果最好，能达到77.5+。

具体实现代码见run_cv.py

### 模型融合

详情见

[CCF贝壳房产聊天问答匹配高分思路](https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&amp;mid=2247487962&amp;idx=1&amp;sn=91269fcde0d47f8f3899bf77fe34e415&amp;chksm=96c43c1fa1b3b509593b2baed411e57f47d5990b316f6c56a6f7e10802a470b0b3cd55239a78&amp;scene=132#wechat_redirect)

具体实现代码见run_cv_lgb_small.py

采用贝叶斯搜索出lightgbm最优参数，详情见run_bayesopt.py
### 另一种分类思路
[直接将BERT的输出做分类，效果相对会差一点，但比较简单](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)
