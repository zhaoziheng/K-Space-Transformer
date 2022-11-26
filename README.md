# K-Space Transformer for Undersampled MRI Reconstruction

This repository contains the PyTorch implementation of K-Space Transformer.

### Citation
If you use this code for your research or project, please cite:

	@inproceedings{zhao2022kspacetransformer,
	  title={K-Space Transformer for Undersampled MRI Reconstruction},
	  author={Ziheng Zhao, Tianjiao Zhang, Weidi Xie, Yanfeng Wang and Ya Zhang},
	  booktitle={British Machine Vision Conference (BMVC)},
	  year={2022}
	}

### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 1.10.0
* torchvision 0.11.1
* numpy 1.21.6
* scikit-image 0.18.1
* matplotlib 3.4.2
* tqdm

### To Run Our Code
- Train the model
```bash
python main.py --output_dir 'Log' --train_hr_data_path 'xxx/xxx.npy' --train_lr_data_path 'xxx/xxx.npy' --train_mask_path 'xxx/xxx.npy' --valid_hr_data_path 'xxx/xxx.npy' --valid_lr_data_path 'xxx/xxx.npy' --valid_mask_path 'xxx/xxx.npy'
```

- Test the model
```bash
python test.py --output_dir 'Record_Path' --modelPath 'xxx/checkpoint.pth' --hr_data_path 'xxx/xxx.npy' --lr_data_path 'xxx/xxx.npy' --mask_path 'xxx/xxx.npy'