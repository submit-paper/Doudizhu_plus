# Doudizhu_plus
This work is based on the DouZero library, so the requirements are the same as DouZero. For convenience, code of original DouZero is also included in the project. Our modifications are listed in other folders and anyone who wants to refer to this job can refer to corresponding code. The installation and other instructons please refer to DouZero(https://github.com/kwai/DouZero). We just talk about changes we make in this project.


## Opponent Modeling
he modification is in the folder. If you want to realize this part, just refer to the code and most description is included in the notes. The test part is a little diferent from DouZero as the models for test are combined with opponent modeling. So we also make modifications is the ``evaluation'' folder. However, the way for operation is the same as DouZero.

## Coach Network
The modification is in the folder. If you want to realize this part, just refer to the code and most description is included in the notes. We only introduce the test code here. We use the files in the ``test'' folder which use os.system to execute relative commands. Every time new models are saved, a new round of test will be started.

## Combination
As opponent modeling and coach network works at different phases of the DouDzihu AI system. In this way, just make corresponding modifications as the two parts.

##Bidding
The modification is in the folder and a README.md file is also included. Please refer to it for more information.
