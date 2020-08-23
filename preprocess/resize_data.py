import argparse
import pdb
import cv2
import os


def main(args):
	train_data = os.listdir(args.raw_train)
	valid_data = os.listdir(args.raw_valid)

	max_w = 500
	new_h = 64

	for i, image in enumerate(valid_data):
		print(i, image)
		if image == "label.txt":
			continue
		im = cv2.imread(os.path.join(args.raw_valid, image))
		h, w, d = im.shape
		unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
		if unpad_im.shape[1] > max_w:
			print(image)
			pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
		else:
			pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite(os.path.join(args.unpad_valid, image), unpad_im)
		cv2.imwrite(os.path.join(args.pad_valid, image), pad_im)

	for i, image in enumerate(train_data):
		print(i, image)
		if image == "label.txt":
			continue
		im = cv2.imread(os.path.join(args.raw_train, image))
		h, w, d = im.shape
		unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
		if unpad_im.shape[1] > max_w:
			pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
		else:
			pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite(os.path.join(args.unpad_train, image), unpad_im)
		cv2.imwrite(os.path.join(args.pad_train, image), pad_im)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--raw_train")
	parser.add_argument("--raw_valid")
	parser.add_argument("--pad_train")
	parser.add_argument("--pad_valid")
	parser.add_argument("--unpad_train")
	parser.add_argument("--unpad_valid")
	args = parser.parse_args()
	main(args)

"""
	python3 resize_data.py --raw_train=data/process_extract_0606/extract_0606_resident/labeled/v2/train/ --raw_valid=data/process_extract_0606/extract_0606_resident/labeled/v2/valid/ --pad_train=data/process_extract_0606/extract_0606_resident/labeled/v2/pad_train/ --pad_valid=data/process_extract_0606/extract_0606_resident/labeled/v2/pad_valid/ --unpad_train=data/process_extract_0606/extract_0606_resident/labeled/v2/unpad_train/ --unpad_valid=data/process_extract_0606/extract_0606_resident/labeled/v2/unpad_valid/

"""
