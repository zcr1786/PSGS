### Installation
```
- torch >= 1.0.0
- sklearn >= 0.21.0
- rdkit.Chem 
```
### Run code

- For single-task classification (such as LogP, FDA, BBBP, BACE datasets):  
    ```
    python PSGS_single/main.py
    ```
- For multi-task classification (such as Tox21 and ToxCast datasets):  
    ```
    python PSGS_multi/main.py
    ```
