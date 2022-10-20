### Installation
```
- torch >= 1.0.0
- sklearn >= 0.21.0
- rdkit.Chem 
```
### Run code

- For single-task classification (such as LogP, FDA, BBBP, BACE datasets):  
    ```
    python GraSeq_single/main.py
    ```
- For multi-task classification (such as Tox21 and ToxCast datasets):  
    ```
    python GraSeq_multi/main.py
    ```
