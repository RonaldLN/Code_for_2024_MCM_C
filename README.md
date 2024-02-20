# Code_for_2024_MCM_C

A repo for my code solving Problem C of 2024 MCM.

# [参赛回忆记录](https://ronaldln.github.io/MyPamphlet-Blog/2024/02/12/2024/)

>   # 2024数模美赛参赛纪实(编程手视角)
>
>   >   2024-02-02 --- 2024-02-06
>
>   ```txt
>       比赛时间为 2号早上6点公布题目，6号早上9点开始提交论文(北京时间)
>   ```
>
>   ## Day 1
>
>   ### 阅读题目 (早)
>
>   早上8点起来，6点起来的队友已经将每个题目都大致看了一遍，然后开始边吃早餐边看题目边和队友讨论题目。
>
>   [...<br/>继续阅读](https://ronaldln.github.io/MyPamphlet-Blog/2024/02/12/2024/)

*记录*对应的代码版本为 `original` 分支中的代码。

# 说明

## 环境安装

```bash
pip install -r requirements.txt
```

## python文件说明

-   `GRU_test_*.py` 用于测试使用TensorFlow GRU模型
-   `final_*.py` 用于对决赛冠军的数据进行训练和预测
-   `all_data_process_step[n]_*.py` 用于处理整个赛季的数据，分成3步 `step1` `step2` `step3`
    -   `public` 为处理第1轮16场*32进16*的比赛的32名选手的数据
    -   `champion` 为处理赛季冠军 **Carlos Alcaraz** 5轮比赛数据
    -   `2nd` 为处理赛季亚军 **Novak Djokovic** 5轮比赛数据
    -   `3rd` 为处理赛季4强 **Jannik Sinner** 4轮比赛数据
-   `count_*.py` 用于计算某些比赛的长度
-   `season_*.py` 用于对整个赛季处理出的数据进行训练和预测
-   `analyse_compare_precise_predictions.py` 用于对精确预测和不精确预测的输入数据进行分析
-   `code_sample.py` 为最后用于论文展示的示例代码

