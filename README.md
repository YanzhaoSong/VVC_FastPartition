# Efficient QTMT Partitioning for VVC Intra-Frame Coding via Multi-Model Fusion

This is the source code for the paper **Efficient QTMT Partitioning for VVC Intra-Frame Coding via Multi-Model Fusion**

## Dataset Download
* CNN dataset: https://pan.baidu.com/s/1K52HIhc6FwPvKmPzLRJQRA (code: cd5c)
* LGBM dataset: https://pan.baidu.com/s/1RQ1xkmy1t93i8i7U_LVZdQ (code: gfag)

## Folder Instruction
* codec: Include the source files and exe of VTM-13.0 implemented with the proposed fast algorithm.
* encoding_test_v13: Include test scripts for the performance of the proposed fast algorithm.
* src: Include files for CNN and LGBM model training.
* test_sequences: For placing test sequences.

## How to train CNN models
1. Indicate the following three directories that hold CNN datasets:  
    ```
    # cnn_engin.py
    self.data_args.root_dir = './'  # root directory for storing CNN dataset

    # load_data_cnn.py
    img_dir = os.path.join(root_dir, 'images')  # directory for storing image data
    pickle_dir = os.path.join(root_dir, 'pickles')  # directory for storing pickles 
    ```
2. Setting the GPU, log path, and training parameters:  
    ```
    # cnn_engine.py
    self.basic_args.date = "1210" 
    self.basic_args.log_index = '0'  
    # used to determine the path to store training data, models
    # for example: log/1210/MyNet/0/
    ```
3. Run `cnn_engine.py`

## How to train LGBM models
1. Indicate the path where the LGBM dataset and models are stored
    ```
    # train_lgbm.ipynb
    pkl_dir = "parquets_lgbm"  # directory for storing parquets
    save_dir = "scripts/lgbm_scripts"  # directory for storing LGBM models
    ```
2. Run `train_lgbm.ipynb`

## How to test encoding performance
You can use the script ``test_script.bat`` under ``encoding_test_v13`` folder.
