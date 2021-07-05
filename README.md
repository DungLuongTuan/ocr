# Tensorflow 2 implimentation of Attention-OCR
## Dataset
Follow the construction of FSNS datasets with 2 folders: ```train``` and ```valid```, each contains images and a ```label.txt``` file
Sample of ```label.txt```:
```
1.jpg Tuan
2.jpg Luong
3.jpg Dungz
```
Column 1 is file names and column 2 is the label of corresponding file

## Preprocess
Resize and pad all images to the same size (change target image width and height in ```resize_data.py``` line 11, 12)
```sh
cd preprocess
python3 resize_data.py --raw_train=path_to_train_folder --raw_valid=path_to_valid_folder --pad_train=path_to_pad_train_folder --pad_valid=path_to_pad_valid_folder --unpad_train=path_to_unpad_train_folder --unpad_valid=path_to_unpad_valid_folder
```

Generate tfRecord
Change parameters in ```gen_record.py```, including ```null_id, max_seqlen, image_width``` (line 41, 42, 77)
```sh
python3 gen_record.py --pad_path=path_to_pad_train_folder --unpad_path=path_to_unpad_train_folder --charset_path=path_to_charset_file --out_path=out_record_path
python3 gen_record.py --pad_path=path_to_pad_valid_folder --unpad_path=path_to_unpad_valid_folder --charset_path=path_to_charset_file --out_path=out_record_path
```
Change parameters in ```hparams.py```
```python
self.train_record_path=path_to_train_record
self.num_train_sample=number_sample_in_train_record
self.valid_record_path=path_to_valid_record
self.num_valid_sample=number_sample_in_valid_record
self.charset_path=path_to_charset_file
self.save_path=save_path
```

## Train new model
```python
python3 train.py
```

View training progress by run tensorboard in ```logs``` folder in ```save_path```
