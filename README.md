# Large language model Papers

## Pretrain
### 2024

## SFT
### 2024

## RLHF
### 2024
- [A Dense Reward View on Aligning Text-to-Image Diffusion with Preference](https://arxiv.org/pdf/2402.08265)
  - Shentao Yang, Tianqi Chen, Mingyuan Zhou
  - Keyword: RLHF for Text-to-Image Generation, Dense Reward Improvement of DPO, Efficient Alignment
  - Code: [Official](https://github.com/Shentao-YANG/Dense_Reward_T2I)

- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/pdf/2401.01335)
  - Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu
  - Keyword: Self-Play Fine-Tuning
  - Code: [Official](https://github.com/uclaml/SPIN)

- [RLHF Deciphered: A Critical Analysis of Reinforcement Learning from Human Feedback for LLMs](https://arxiv.org/abs/2404.08555)
  - Shreyas Chaudhari, Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, Karthik Narasimhan, Ameet Deshpande, Bruno Castro da Silva
  - Keyword: RLHF, Oracular Reward, Reward Model Analysis, Survey 

- [Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases](https://arxiv.org/abs/2402.08552)
  - Ziyi Zhang, Sen Zhang, Yibing Zhan, Yong Luo, Yonggang Wen, Dacheng Tao
  - Keyword: Diffusion Models, Alignment, Reinforcement Learning, RLHF, Reward Overoptimization, Primacy Bias
  - Code: [official](https://github.com/ZiyiZhang27/tdpo)

- [On Diversified Preferences of Large Language Model Alignment](https://arxiv.org/pdf/2312.07401.pdf)
  - Dun Zeng, Yong Dai, Pengyu Cheng, Tianhao Hu, Wanshun Chen, Nan Du, Zenglin Xu
  - Keyword: Aligning shared preference, Reward modeling metrics, LLM
  - Code: [official](https://github.com/dunzeng/MORE)

- [Aligning Crowd Feedback via Distributional Preference Reward Modeling](https://arxiv.org/pdf/2402.09764.pdf)
  - Dexun Li, Cong Zhang, Kuicai Dong, Derrick Goh Xin Deik, Ruiming Tang, Yong Liu
  - Keyword: RLHF, Preference distribution, Aligning, LLM

- [Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization](https://arxiv.org/pdf/2310.03708.pdf)
  - Zhanhui Zhou, Jie Liu, Chao Yang, Jing Shao, Yu Liu, Xiangyu Yue, Wanli Ouyang, Yu Qiao
  - Keyword: Multi-objective RLHF without reward modeling, DPO
  - Code: [official](https://github.com/ZHZisZZ/modpo/)
  
- [Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!](https://arxiv.org/pdf/2402.12343.pdf)
  - Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao
  - Keyword: LLM inference-time attack, DPO, Producing harmful LLMs without training
  - Code: [official](https://github.com/ZHZisZZ/emulated-disalignment/)
 
- [A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference](https://arxiv.org/pdf/2402.07314.pdf)
  - Chenlu Ye, Wei Xiong, Yuheng Zhang, Nan Jiang, Tong Zhang
  - Keyword: Game-based RLHF, Nash Learning, Alignment under reward-model-free oracle

- [Mitigating the Alignment Tax of RLHF](https://arxiv.org/pdf/2309.06256.pdf)
  - Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, Hanze Dong, Renjie Pi, Han Zhao, Nan Jiang, Heng Ji, Yuan Yao, Tong Zhang
  - Keyword: RLHF, Alignment tax, Catastrophic forgetting 

- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/pdf/2305.13301.pdf)
  - Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine
  - Keyword: reinforcement learning, RLHF, diffusion models
  - Code: [official](http://rl-diffusion.github.io/)

- [AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model](https://openreview.net/forum?id=bxfKIYfHyx)
  - Zibin Dong, Yifu Yuan, Jianye Hao, Fei Ni, Yao Mu, Yan Zheng,Yujing Hu, Tangjie Lv, Changjie Fan, Zhipeng Hu
  - Keyword: Reinforcement learning; Diffusion models; RLHF; Preference aligning
  - Code: [official](https://aligndiff.github.io/)

- [Dense Reward for Free in Reinforcement Learning from Human Feedback](https://browse.arxiv.org/pdf/2402.00782)
  - Alex J. Chan, Hao Sun, Samuel Holt, Mihaela van der Schaar
  - Keyword: RLHF
  - Code: [official](https://github.com/XanderJC/attention-based-credit)

- [Transforming and Combining Rewards for Aligning Large Language Models](https://browse.arxiv.org/abs/2402.00742)
  - Zihao Wang, Chirag Nagpal, Jonathan Berant, Jacob Eisenstein, Alex D'Amour, Sanmi Koyejo, Victor Veitch
  - Keyword: RLHF, Aligning, LLM

### 2023
- [Preference-grounded Token-level Guidance for Language Model Fine-tuning](https://proceedings.neurips.cc/paper_files/paper/2023/file/4d4a3b6a34332d80349137bcc98164a5-Paper-Conference.pdf)
  - Shentao Yang, Shujian Zhang, Congying Xia, Yihao Feng, Caiming Xiong, Mingyuan Zhou
  - Keyword: RLHF, Token-level Training Guidance, Alternate/Online Training Framework, Minimalist Training Objectives
  - Code: [Official](https://github.com/Shentao-YANG/Preference_Grounded_Guidance)

- [Fantastic Rewards and How to Tame Them: A Case Study on Reward Learning for Task-oriented Dialogue Systems](https://arxiv.org/pdf/2302.10342)
  - Yihao Feng*, Shentao Yang*, Shujian Zhang, Jianguo Zhang, Caiming Xiong, Mingyuan Zhou, Huan Wang
  - Keyword: RLHF, Genralized Reward Function Learning, Reward Function Utilization, Task-oriented Dialogue System, Learning-to-rank
  - Code: [Official](https://github.com/Shentao-YANG/Fantastic_Reward_ICLR2023)

- [Inverse Preference Learning: Preference-based RL without a Reward Function](https://arxiv.org/pdf/2305.15363)
  - Joey Hejna, Dorsa Sadigh
  - Keyword: Inverse Preference Learning, without reward model
  - Code: [Official](https://github.com/jhejna/inverse-preference-learning)

- [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf)
  - Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S. Liang, Tatsunori B. Hashimoto
  - Keyword: RLHF, Simulation Framework
  - Code: [Official](https://github.com/tatsu-lab/alpaca_farm)

- [Preference Ranking Optimization for Human Alignment](https://arxiv.org/pdf/2306.17492)
  - Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, Houfeng Wang
  - Keyword: Preference Ranking Optimization
  - Code: [Official](github.com/AlibabaResearch/DAMO-ConvAI/tree/main/PRO)

- [Adversarial Preference Optimization](https://arxiv.org/abs/2311.08045)
  - Pengyu Cheng, Yifan Yang, Jian Li, Yong Dai, Nan Du
  - Keyword: RLHF, GAN, Adversarial Games
  - Code: [Official](https://github.com/Linear95/APO)

- [Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint](https://arxiv.org/abs/2312.11456)
  - Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, Tong Zhang
  - Keyword: RLHF, Iterative DPO, Mathematical foundation

- [Sample Efficient Reinforcement Learning from Human Feedback via Active Exploration](https://arxiv.org/abs/2312.00267)
  - Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, Willie Neiswanger
  - Keyword: RLHF, sample efficience, exploration

- [Reinforcement Learning from Statistical Feedback: the Journey from AB Testing to ANT Testing](https://arxiv.org/abs/2311.14766)
  - Feiyang Han, Yimin Wei, Zhaofeng Liu, Yanxing Qi
  - Keyword: RLHF, AB testing, RLSF

- [A Baseline Analysis of Reward Models' Ability To Accurately Analyze Foundation Models Under Distribution Shift](https://arxiv.org/abs/2311.14743)
  - Ben Pikus, Will LeVine, Tony Chen, Sean Hendryx
  - Keyword: RLHF, OOD, Distribution Shift 

- [Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language](https://arxiv.org/abs/2311.14543)
  - Di Jin, Shikib Mehri, Devamanyu Hazarika, Aishwarya Padmakumar, Sungjin Lee, Yang Liu, Mahdi Namazifar
  - Keyword: RLHF, data-efficient, Alignment

- [Let's Reinforce Step by Step](https://arxiv.org/abs/2311.05821)
  - Sarah Pan, Vladislav Lialin, Sherin Muckatira, Anna Rumshisky
  - Keyword: RLHF, reasoning

- [Direct Preference-based Policy Optimization without Reward Modeling](https://arxiv.org/abs/2301.12842)
  - Gaon An, Junhyeok Lee, Xingdong Zuo, Norio Kosaka, Kyung-Min Kim, Hyun Oh Song
  - Keyword: RLHF without reward modeling, Contrastive learning, Offline refinforcement learning

- [AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model](https://arxiv.org/abs/2310.02054)
  - Zibin Dong, Yifu Yuan, Jianye Hao, Fei Ni, Yao Mu, Yan Zheng, Yujing Hu, Tangjie Lv, Changjie Fan, Zhipeng Hu
  - Keyword: RLHF, Alignment, Diffusion model

- [Eureka: Human-Level Reward Design via Coding Large Language Models](https://arxiv.org/abs/2310.12931)
  - Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, Anima Anandkumar
  - Keyword: LLM based, reward functions design

- [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773)
  - Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, Yaodong Yang
  - Keyword: Sale RL, LLM fine-ture

- [Quality Diversity through Human Feedback](https://arxiv.org/abs/2310.12103)
  - Li Ding, Jenny Zhang, Jeff Clune, Lee Spector, Joel Lehman
  - Keyword: Quality Diversity, Diffusion model

- [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505)
  - Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, Zhi-Quan Luo
  - Keyword: computational efficiency, variance-reduction technique

- [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242.pdf)
  - André Susano Pinto, Alexander Kolesnikov, Yuge Shi, Lucas Beyer, Xiaohua Zhai
  - Keyword: Reward tuning in Computer Vision

- [The Wisdom of Hindsight Makes Language Models Better Instruction Followers](https://arxiv.org/pdf/2302.05206.pdf)
  - Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, Joseph E. Gonzalez
  - Keyword: Hindsight Instruction Relabeling, RLHF System, No Value Network Required
  - Code: [official](https://github.com/tianjunz/HIR)

- [Language Instructed Reinforcement Learning for Human-AI Coordination](https://arxiv.org/pdf/2304.07297.pdf)
  - Hengyuan Hu, Dorsa Sadigh
  - Keyword: Human-AI coordination, Human preference alignment, Instruction conditioned RL

- [Aligning Language Models with Offline Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2308.12050.pdf)
  - Jian Hu, Li Tao, June Yang, Chandler Zhou
  - Keyword: Decision Transformer-based Alignment, Offline Reinforcement Learning, RLHF System

- [Preference Ranking Optimization for Human Alignment](https://arxiv.org/pdf/2306.17492.pdf)
  - Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li and Houfeng Wang
  - Keyword: Supervised Human Preference Alignment, Preference Ranking Extension
  - Code: [official](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/PRO)

- [Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation](https://arxiv.org/abs/2305.00955)
  - Patrick Fernandes, Aman Madaan, Emmy Liu, António Farinhas, Pedro Henrique Martins, Amanda Bertsch, José G. C. de Souza, Shuyan Zhou, Tongshuang Wu, Graham Neubig, André F. T. Martins
  - Keyword: Natural Language Generation, Human Feedback Integration, Feedback Formalization and Taxonomy, AI Feedback and Principles-Based Judgments

- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
  - OpenAI
  - Keyword: A large-scale, multimodal model, Transformerbased model, Fine-tuned used RLHF
  - Code: [official](https://github.com/openai/evals)
  - Dataset: [DROP](https://allenai.org/data/drop), [WinoGrande](https://winogrande.allenai.org/), [HellaSwag](https://rowanzellers.com/hellaswag/), [ARC](https://allenai.org/data/arc), [HumanEval](https://github.com/openai/human-eval), [GSM8K](https://paperswithcode.com/dataset/gsm8k), [MMLU](https://paperswithcode.com/dataset/mmlu), [TruthfulQA](https://github.com/sylinrl/TruthfulQA)

- [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment](https://arxiv.org/pdf/2304.06767.pdf)
  - Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, Tong Zhang
  - Keyword: Rejection Sampling Finetuning, Alternative to PPO, Diffusion Model
  - Code: [official](https://github.com/OptimalScale/LMFlow)
  
- [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/pdf/2304.05302v1.pdf)
  - Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang
  - Keyword: New paradigm for RLHF
  - Code: [official](https://github.com/GanjinZero/RRHF)

- [Few-shot Preference Learning for Human-in-the-Loop RL](https://openreview.net/pdf?id=IKC5TfXLuW0)
  - Joey Hejna, Dorsa Sadigh
  - Keyword: Preference Learning, Interactive Learning, Multi-task Learning, Expanding the pool of available data by viewing human-in-the-loop RL
  - Code: [official](https://github.com/jhejna/few-shot-preference-rl)

- [Better Aligning Text-to-Image Models with Human Preference](https://arxiv.org/abs/2303.14420)
  - Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, Hongsheng Li
  - Keyword: Diffusion Model, Text-to-Image, Aesthetic
  - Code: [official](https://github.com/tgxs002/align_sd)

- [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/pdf/2304.05977v2.pdf)
  - Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong
  - Keyword: General-purpose text-to-Image human preference RM, Evaluating Text-to-Image Generative Models
  - Code: [official](https://github.com/THUDM/ImageReward)
  - Dataset: [COCO](https://cocodataset.org/#home), [DiffusionDB](https://poloclub.github.io/diffusiondb/)

- [Aligning Text-to-Image Models using Human Feedback](https://arxiv.org/pdf/2302.12192.pdf)
  - Kimin Lee, Hao liu, MoonKyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Shixiang Shane Gu
  - Keyword: Text-to-Image, Stable diffusion model, Reward function that predicts human feedback

- [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models](https://arxiv.org/pdf/2303.04671.pdf)
  - Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, Nan Duan
  - Keyword: Visual Foundation Models, Visual ChatGPT 
  - Code: [official](https://github.com/microsoft/visual-chatgpt)

- [Pretraining Language Models with Human Preferences](https://arxiv.org/abs/2302.08582) (PHF)
  - Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Bhalerao, Christopher L. Buckley, Jason Phang, Samuel R. Bowman, Ethan Perez
  - Keyword: Pretraining, offline RL, Decision transformer
  - Code: [official](https://github.com/tomekkorbak/pretraining-with-human-feedback)

- [Aligning Language Models with Preferences through f-divergence Minimization](https://arxiv.org/abs/2302.08215) (f-DPG)
  - Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, Marc Dymetman
  - Keyword: f-divergence, RL with KL penalties

- [Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons](https://arxiv.org/pdf/2301.11270.pdf)
  - Banghua Zhu, Jiantao Jiao, Michael I. Jordan
  - Keyword: Pessimistic MLE, Max-entropy IRL

- [The Capacity for Moral Self-Correction in Large Language Models](https://arxiv.org/pdf/2302.07459.pdf)
  - Anthropic
  - Keyword: Improve moral self-correction capability by increasing RLHF training
  - Dataset; [BBQ](https://github.com/nyu-mll/BBQ)

### 2022

- [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241) (NLPO)
  - Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kianté,Brantley, Jack Hessel, Rafet Sifa, Christian Bauckhage, Hannaneh Hajishirzi, Yejin Choi
  - Keyword: Optimizing language generators with RL, Benchmark,  Performant RL algorithm
  - Code: [official](https://github.com/allenai/RL4LMs)
  - Dataset: [IMDB](https://www.imdb.com/interfaces/), [CommonGen](https://inklab.usc.edu/CommonGen/), [CNN Daily Mail](https://github.com/abisee/cnn-dailymail), [ToTTo](https://github.com/google-research-datasets/ToTTo), [WMT-16 (en-de)](https://www.statmt.org/wmt16/it-translation-task.html),[NarrativeQA](https://github.com/deepmind/narrativeqa), [DailyDialog](http://yanran.li/dailydialog)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
  - Leo Gao, John Schulman, Jacob Hilton
  - Keyword: Gold reward model train proxy reward model, Dataset size, Policy parameter size, BoN, PPO
- [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/abs/2209.14375) (Sparrow)
  - Amelia Glaese, Nat McAleese, Maja Trębacz, et al.
  - Keyword: Information-seeking dialogue agent, Break down the good dialogue into natural language rules, DPC, Interact with the model to elicit violation of a specific rule (Adversarial Probing)
  - Dataset: [Natural Questions](https://ai.google.com/research/NaturalQuestions), [ELI5](https://facebookresearch.github.io/ELI5/), [QuALITY](https://github.com/nyu-mll/quality), [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [WinoBias](https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino), [BBQ](https://github.com/nyu-mll/BBQ)
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)
  - Deep Ganguli, Liane Lovitt, Jackson Kernion, et al.
  - Keyword: Red team language model, Investigate scaling behaviors, Read teaming Dataset
  - Code: [official](https://github.com/anthropics/hh-rlhf)
- [Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning](https://arxiv.org/abs/2208.02294)
  - Deborah Cohen, Moonkyung Ryu, Yinlam Chow, Orgad Keller, Ido Greenberg, Avinatan Hassidim, Michael Fink, Yossi Matias, Idan Szpektor, Craig Boutilier, Gal Elidan
  - Keyword: Real-time, Open-ended dialogue system, Pairs the succinct embedding of the conversation state by language models, CAQL, CQL, [BERT](https://github.com/google-research/bert)
- [Quark: Controllable Text Generation with Reinforced Unlearning](https://arxiv.org/abs/2205.13636)
  - Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithviraj Ammanabrolu, Yejin Choi
  - Keyword: Fine-tuning the language model on signals of what not to do, Decision Transformer, LLM tuning with PPO
  - Code: [official](https://github.com/gximinglu/quark)
  - Dataset: [WRITINGPROMPTS](https://www.kaggle.com/datasets/ratthachat/writing-prompts), [SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), [WIKITEXT-103](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
  - Yuntao Bai, Andy Jones, Kamal Ndousse, et al.
  - Keyword: Harmless assistants, Online mode, Robustness of RLHF training, OOD detection.
  - Code: [official](https://github.com/anthropics/hh-rlhf)
  - Dataset: [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [HellaSwag](https://rowanzellers.com/hellaswag/), [ARC](https://allenai.org/data/arc), [OpenBookQA](https://allenai.org/data/open-book-qa), [LAMBADA](https://zenodo.org/record/2630551#.Y_KLJ-yZNhF), [HumanEval](https://github.com/openai/human-eval), [MMLU](https://github.com/hendrycks/test), [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [Teaching language models to support answers with verified quotes](https://arxiv.org/abs/2203.11147) (GopherCite)
  - Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Geoffrey Irving, Nat McAleese
  - Keyword: Generate answers which citing specific evidence, Abstain from answering when unsure
  - Dataset: [Natural Questions](https://ai.google.com/research/NaturalQuestions), [ELI5](https://facebookresearch.github.io/ELI5/), [QuALITY](https://github.com/nyu-mll/quality), [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)
  - Long Ouyang, Jeff Wu, Xu Jiang, et al.
  - Keyword: Large Language Model, Align Language Model with Human Intent
  - Code: [official](https://github.com/openai/following-instructions-human-feedback)
  - Dataset: [TruthfulQA](https://github.com/sylinrl/TruthfulQA), [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf)
  - Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, et al.
  - Keyword: RL from AI feedback(RLAIF), Training a harmless AI assistant through selfimprovement, Chain-of-thought style, Control AI behavior more precisely
  - Code: [official](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)
- [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)
  - Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, Edwin Chen, et al.
  - Keyword: Automatically generate evaluations with LMs, More RLHF makes LMs worse, LM-written evaluations are highquality
  - Code: [official](https://github.com/anthropics/evals)
  - Dataset: [BBQ](https://github.com/nyu-mll/BBQ), [Winogender Schemas](https://github.com/rudinger/winogender-schemas)
- [Non-Markovian Reward Modelling from Trajectory Labels via Interpretable Multiple Instance Learning](https://arxiv.org/abs/2205.15367)
  - Joseph Early, Tom Bewley, Christine Evers, Sarvapali Ramchurn
  - Keyword: Reward Modelling (RLHF), Non-Markovian, Multiple Instance Learning, Interpretability
  - Code: [official](https://github.com/JAEarly/MIL-for-Non-Markovian-Reward-Modelling)

### 2021
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (WebGPT)
  - Reiichiro Nakano, Jacob Hilton, Suchir Balaji, et al.
  - Keyword: Model search the web and provide reference， Imitation learning， BC, long form question
  - Dataset: [ELI5](https://facebookresearch.github.io/ELI5/), [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862)
  - Jeff Wu, Long Ouyang, Daniel M. Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, Paul Christiano
  - Keyword:  Model trained on small task to assist human evaluate broader task, BC
  - Dataset: [Booksum](https://github.com/salesforce/booksum), [NarrativeQA](https://github.com/deepmind/narrativeqa)
- [Revisiting the Weaknesses of Reinforcement Learning for Neural Machine Translation](https://arxiv.org/abs/2106.08942)
  - Samuel Kiegeland, Julia Kreutzer
  - Keyword:  The success of policy gradient is because of reward rather than the shape of output distribution, Machine Translation, NMT, DOmain Adaption
  - Code: [official](https://github.com/samuki/reinforce-joey)
  - Dataset: [WMT15](https://www.statmt.org/wmt15/index.html), [IWSLT14](https://sites.google.com/site/iwsltevaluation2014/mt-track)

### 2020 and before

- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
  - Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano
  - Keyword: Care about summary quality, Training loss affect the model behavior, Reward model generalizes to new datasets
  - Code: [official](https://github.com/openai/summarize-from-feedback)
  - Dataset: [TL;DR](https://www.tensorflow.org/datasets/catalog/reddit), [CNN/DM](https://github.com/abisee/cnn-dailymail)
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
  - Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving
  - Keyword: Reward learning for language, Continuing text with positive sentiment, Summary task, Physical descriptive
  - Code: [official](https://github.com/openai/lm-human-preferences)
  - Dataset: [TL;DR](https://www.tensorflow.org/datasets/catalog/reddit), [CNN/DM](https://github.com/abisee/cnn-dailymail)
- [Scalable agent alignment via reward modeling: a research direction](https://arxiv.org/abs/1811.07871)
  - Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, Shane Legg
  - Keyword: Agent alignment problem, Learn reward from interaction, Optimize reward with RL, Recursive reward modeling
  - Code: [official](https://github.com/rddy/ReQueST)
  - Env: Atari
- [Reward learning from human preferences and demonstrations in Atari](https://arxiv.org/abs/1811.06521)
  - Borja Ibarz, Jan Leike, Tobias Pohlen, Geoffrey Irving, Shane Legg, Dario Amodei
  - Keyword: Expert demonstration trajectory preferences reward hacking problem, Noise in human label
  - Code: [official](https://github.com/rddy/ReQueST)
  - Env: Atari
- [Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces](https://arxiv.org/abs/1709.10163)
  - Garrett Warnell, Nicholas Waytowich, Vernon Lawhern, Peter Stone
  - Keyword:  High dimension state, Leverage the input of Human trainer
  - Code: [third party](https://github.com/bharadwaj1098/Tamer)
  - Env: Atari
- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
  - Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei
  - Keyword: Explore goal defined in human preferences between pairs of trajectories segmentation, Learn more complex thing than human feedback
  - Code: [official](https://github.com/mrahtz/learning-from-human-preferences)
  - Env: Atari, MuJoCo
- [Interactive Learning from Policy-Dependent Human Feedback](https://arxiv.org/abs/1701.06049)
  - James MacGlashan, Mark K Ho, Robert Loftin, Bei Peng, Guan Wang, David Roberts, Matthew E. Taylor, Michael L. Littman
  - Keyword: Decision is influenced by current policy rather than human feedback, Learn from policy dependent feedback that converges to a local optimal

  
## Agent
### 2024

## MLLM
### 2024
