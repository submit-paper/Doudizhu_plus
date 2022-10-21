# Completed DouZero+
This work is based on the DouZero library, so the requirements are the same as DouZero. For convenience, code of original DouZero is also included in the project. Our modifications are listed in other folders and anyone who wants to refer to this job can refer to corresponding code. The installation and other instructons please refer to DouZero(https://github.com/kwai/DouZero). We just talk about changes we make in this project.

The Completed DouZero+ is composed of three parts, namely, simulate, train and evaluate. 
In the 'simulate' part, we add a folder called 'bid' and the 'evaluation.py' file is the one that matters. Also, the record.py file is the file that is used when executing simulation. Other files are the same as DouZero+ and there's need to pay attention. The usage is the same as evaluation of DouZero, that is, you just need set the models and the simulation times that you want to use.

The 'train' part contains the training files. After collecting enough data samples, you can train your bidding network. As supervised learning is adopted, it is not complex to understand. After training a model, grid search can be used to determine the parameters for bidding phase as there are only three risk stakes.

The 'evaluate' part is the code that we use in the evaluation experiments in the article. The 'douzero/env/game.py' and 'douzero/bid/evaluation.py' file is changed and we add a 'douzero/bid/bid_agent.py'. The game.py construct the game environment for DouDizhu and you can choose which bidding method is used in this file. The evaluation.py records the evaluation results. Other remaining files are also unchanged.
