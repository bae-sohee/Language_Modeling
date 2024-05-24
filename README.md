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


- 모델 별 생성 결과 비교 (seed별 모델 생성 결과)  
## Generated Text Samples

여기에는 다양한 시드 문자열을 사용하여 생성된 텍스트 샘플을 LSTM과 RNN 모델로 비교한 결과를 제공합니다.

| Seed     | LSTM                                                                                         | RNN                                                                                          |
|----------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| ROMEO:   | 텍스트 예시 1 - LSTM                                                                          | 텍스트 예시 1 - RNN                                                                          |
| JULIET:  | 텍스트 예시 2 - LSTM                                                                          | 텍스트 예시 2 - RNN                                                                          |
| MACBETH: | 텍스트 예시 3 - LSTM                                                                          | 텍스트 예시 3 - RNN                                                                          |
| HAMLET:  | 텍스트 예시 4 - LSTM                                                                          | 텍스트 예시 4 - RNN                                                                          |
| OTHELLO: | 텍스트 예시 5 - LSTM                                                                          | 텍스트 예시 5 - RNN                                                                          |
| Example: | "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (LSTM)| "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (RNN)|


- temperature별 모델 생성 결과 및 해석 (다양한 온도를 시도하고, 온도가 어떤 차이를 만드는지, 왜 더 그럴듯한 결과를 생성하는 데 도움이 되는지에 대해 논의)  
## Generated Text Samples by Temperature

여기에는 다양한 온도 설정에서 생성된 텍스트 샘플을 LSTM과 RNN 모델로 비교한 결과를 제공합니다.

| Temperature | LSTM                                                                                          | RNN                                                                                           |
|-------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 0.5         | 텍스트 예시 (온도 0.5) - LSTM                                                                  | 텍스트 예시 (온도 0.5) - RNN                                                                  |
| 1.0         | 텍스트 예시 (온도 1.0) - LSTM                                                                  | 텍스트 예시 (온도 1.0) - RNN                                                                  |
| 1.5         | 텍스트 예시 (온도 1.5) - LSTM                                                                  | 텍스트 예시 (온도 1.5) - RNN                                                                  |
| 2.0         | 텍스트 예시 (온도 2.0) - LSTM                                                                  | 텍스트 예시 (온도 2.0) - RNN                                                                  |
| Example:    | "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (LSTM)| "ROMEO: What light through yonder window breaks? It is the east, and Juliet is the sun." (RNN)|
- temperature의 결과에 따라서 결과 작성  
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
