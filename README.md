# Language Modeling
- Building a 'many-to-many' recurrent neural network language model with the Shakespeare dataset.

### Assignment Objective
- Vanilla RNN and LSTM models are implemented to compare their performance.
- Analyzing the quality of text generated using various temperature parameters

## 1. Environment
- Python version is 3.8.
- Used 'PyTorch' and device type as 'GPU'.
- `requirements.txt` file is required to set up the virtual environment for running the program. This file contains a list of all the libraries needed to run your program and their versions.

    #### In **Anaconda** Environment,

  ```
  $ conda create -n [your virtual environment name] python=3.8
  
  $ conda activate [your virtual environment name]
  
  $ pip install -r requirements.txt
  ```

  - Create your own virtual environment.
  - Activate your Anaconda virtual environment where you want to install the package. If your virtual environment is named 'test', you can type **conda activate test**.
  - Use the command **pip install -r requirements.txt** to install libraries.

## 2. Dataset
- Run `dataset.py` to determine the length of the dataset.

  ```bash
  python dataset.py
  ```

  ```bash
  shakespeare.txt

     First Citizen:
     Before we proceed any further, hear me speak.

     All:
     Speak, speak.

     First Citizen:
     You are all resolved rather to die than to famish?

     All:
     Resolved. resolved.
     ...
  ```

## 3. Implementation
- You need to run `main.py`.  
  Training using the LSTM model:  

  ```bash
  python main.py --lstm
  ```  

  Training using the RNN model:  
  
  ```bash
  python main.py --lstm
  ```  

- The default settings are as follows.
    #### setting
    epoch = 200  
    batch_size = 64
    hidden_size = 256
    optimizer = AdamW
    lr = 0.0001  

- You need to run `generate.py`.  
  Create text with LSTM model:  

  ```bash
  python generate.py --lstm
  ```  

  Create text with RNN model:  
  
  ```bash
  python main.py --lstm
  ```  

- The default settings are as follows.
    #### args
    seed = 'ROMEO: '  
    temperature = 1.0
    length = 100
    output = 'generated_text.txt'  

## 4. Result

- Loss for each model
  - LSTM  
  ![loss_plot_LSTM](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/e0be35c8-21dd-4be8-929a-e36f9f793ed8)  
  - RNN  
  ![loss_plot_RNN](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/a2b00775-09ff-40bd-85d4-eaff0f860855)  


- �� �� ���� ��� �� (seed�� �� ���� ���)  
## Generated Text Samples

���⿡�� �پ��� �õ� ���ڿ��� ����Ͽ� ������ �ؽ�Ʈ ������ LSTM�� RNN �𵨷� ���� ����� �����մϴ�.

| Seed     | LSTM                                                                                         | RNN                                                                                          |
|----------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| ROMEO:   | �ؽ�Ʈ ���� 1 - LSTM                                                                          | �ؽ�Ʈ ���� 1 - RNN                                                                          |
| JULIET:  | �ؽ�Ʈ ���� 2 - LSTM                                                                          | �ؽ�Ʈ ���� 2 - RNN                                                                          |
| MACBETH: | �ؽ�Ʈ ���� 3 - LSTM                                                                          | �ؽ�Ʈ ���� 3 - RNN                                                                          |
| HAMLET:  | �ؽ�Ʈ ���� 4 - LSTM                                                                          | �ؽ�Ʈ ���� 4 - RNN                                                                          |
| OTHELLO: | �ؽ�Ʈ ���� 5 - LSTM                                                                          | �ؽ�Ʈ ���� 5 - RNN                                                                          |
| Example: | "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (LSTM)| "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (RNN)|


- temperature�� �� ���� ��� �� �ؼ� (�پ��� �µ��� �õ��ϰ�, �µ��� � ���̸� �������, �� �� �׷����� ����� �����ϴ� �� ������ �Ǵ����� ���� ����)  
## Generated Text Samples by Temperature

���⿡�� �پ��� �µ� �������� ������ �ؽ�Ʈ ������ LSTM�� RNN �𵨷� ���� ����� �����մϴ�.

| Temperature | LSTM                                                                                          | RNN                                                                                           |
|-------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 0.5         | �ؽ�Ʈ ���� (�µ� 0.5) - LSTM                                                                  | �ؽ�Ʈ ���� (�µ� 0.5) - RNN                                                                  |
| 1.0         | �ؽ�Ʈ ���� (�µ� 1.0) - LSTM                                                                  | �ؽ�Ʈ ���� (�µ� 1.0) - RNN                                                                  |
| 1.5         | �ؽ�Ʈ ���� (�µ� 1.5) - LSTM                                                                  | �ؽ�Ʈ ���� (�µ� 1.5) - RNN                                                                  |
| 2.0         | �ؽ�Ʈ ���� (�µ� 2.0) - LSTM                                                                  | �ؽ�Ʈ ���� (�µ� 2.0) - RNN                                                                  |
| Example:    | "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (LSTM)| "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (RNN)|
- temperature�� ����� ���� ��� �ۼ�  
- 



## 5. Refecence

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Dataset  
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 

Model
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
https://data-science-hi.tistory.com/190

Generate
https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
