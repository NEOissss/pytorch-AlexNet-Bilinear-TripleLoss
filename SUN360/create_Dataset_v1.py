import os
import numpy as np
import cv2
import csv
import glob
import random
import json
from sklearn.model_selection import train_test_split

ROOT_PATH = '/mnt/nfs/scratch1/gluo/SUN360/'
CROP_IMG_PATH = ROOT_PATH + 'HalfHalf/IMGs/'


def get_img_hist(part):
    hist_left, hist_right = [], []

    for i in range(max(len(part), 120)):
        im_l = cv2.imread(part[i] + '/L.jpg')
        im_r = cv2.imread(part[i] + '/R.jpg')

        hist_left.append(cv2.calcHist([im_l], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]))
        hist_right.append(cv2.calcHist([im_r], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]))

    return hist_left, hist_right


def generate_dataset(part_A, part_B, train=True):
    if train:
        task_path = ROOT_PATH + 'HalfHalf/task_train_v1'
        gt_path = ROOT_PATH + 'HalfHalf/gt_train_v1.csv'
        print('Start generate half&half training set...')
    else:
        task_path = ROOT_PATH + 'HalfHalf/task_test_v1'
        gt_path = ROOT_PATH + 'HalfHalf/gt_test_v1.csv'
        print('Start generate half&half testing set...')

    num_A = len(part_A)
    num_B = len(part_B)

    hist_A_left, hist_A_right = get_img_hist(part_A)
    hist_B_left, hist_B_right = get_img_hist(part_B)

    print('Anticipating #sample: ', num_A)

    compare_method = 3  # means that the smaller the compareHist, the similar
    num_choice = 10
    K = 1000

    task_ref = []  #
    task_tgt = []  #
    task_choices_minus_1 = []

    for i_ref in range(num_A):
        lr = np.random.randint(2)
        if lr == 0:  # pick left half as the reference image
            task_ref.append(-1*i_ref)  # negative value means ref images are picked from left
            task_tgt.append(i_ref)     # positive value means tgt images are picked from right

            hist_Comp = []
            for j in range(num_B):     # ref in A_left, choices in B_right
                halfi_halfj = cv2.compareHist(hist_A_left[i_ref], hist_B_right[j], compare_method)
                hist_Comp.append(halfi_halfj)

            # the top K in terms of color histogram similarity from the GT target image.
            matchest_topK = np.argpartition(np.array(hist_Comp),K)[:K]
            # randomly sample (#Choices-1) among the top K
            arg_choices_minus_1 = np.random.choice(K, num_choice-1, replace=False)
            choices_minus_1 = matchest_topK[arg_choices_minus_1]
            task_choices_minus_1.append(choices_minus_1)      # choices in B_right, positive

        else:        # pick right half as the reference image
            task_ref.append(i_ref)          # positive value means ref images are picked from right
            task_tgt.append(-1*i_ref)       # negative value means tgt images are picked from left

            hist_Comp = []
            for j in range(num_B):         # ref in A_right, choices in B_left
                halfi_halfj = cv2.compareHist(hist_A_right[i_ref], hist_B_left[j], compare_method)
                hist_Comp.append(halfi_halfj)

            matchest_topK = np.argpartition(np.array(hist_Comp),K)[:K]
            arg_choices_minus_1 = np.random.choice(K, num_choice-1, replace=False)
            choices_minus_1 = matchest_topK[arg_choices_minus_1]
            task_choices_minus_1.append(choices_minus_1 * (-1))  # choices in B_left, negative value

    samples_num = 16  # take samples_num of samples to show the results
    len_sample_id = 9  # e.g 000000666

    gt_res = []

    if not os.path.exists(task_path):
        os.mkdir(task_path)

    for i in range(max(num_A, samples_num)):
        str_i = '0'*(len_sample_id-len(str(i))) + str(i)
        img_list = ['', ['' for i in range(10)]]

        if task_ref[i] < 0 or (task_ref[i] == 0 and sum(task_choices_minus_1[i]) > 0):
            crop_img_path = part_A[i].split('/')[-1] + '/L.jpg'
            img_list[0] = crop_img_path

            gt_id = random.randint(0, num_choice-1)  # randomly set up gt_id from {0, 1, 2, ...., num_choice-1}
            crop_img_path = part_A[i].split('/')[-1] + '/R.jpg'
            img_list[1][gt_id] = crop_img_path

            gt_res.append([str_i, str(gt_id)])

            for choice_i in range(num_choice):
                if choice_i < gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i]].split('/')[-1] + '/R.jpg'
                    img_list[1][choice_i] = crop_img_path
                elif choice_i > gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i-1]].split('/')[-1] + '/R.jpg'
                    img_list[1][choice_i] = crop_img_path

        elif task_ref[i] > 0 or (task_ref[i] == 0 and sum(task_choices_minus_1[i]) < 0):
            crop_img_path = part_A[i].split('/')[-1] + '/R.jpg'
            img_list[0] = crop_img_path

            gt_id = random.randint(0, num_choice-1)  # randomly set up gt_id from {0, 1, 2, ...., num_choice-1}
            crop_img_path = part_A[i].split('/')[-1] + '/L.jpg'
            img_list[1][gt_id] = crop_img_path

            gt_res.append([str_i, str(gt_id)])

            for choice_i in range(num_choice):
                if choice_i < gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i]*(-1)].split('/')[-1] + '/L.jpg'
                    img_list[1][choice_i] = crop_img_path
                elif choice_i > gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i-1]*(-1)].split('/')[-1] + '/L.jpg'
                    img_list[1][choice_i] = crop_img_path

        with open(task_path+'/'+str_i+'.json', 'w') as f:
            json.dump(img_list, f)

        if train:
            print("Training example " + str_i + ' generated')
        else:
            print("Testing example " + str_i + ' generated')

    print(len(gt_res), 'samples generated.')

    with open(gt_path, "w+") as my_csv:            # writing the file as my_csv
        csv_writer = csv.writer(my_csv, delimiter=',')  # using the csv module to write the file
        csv_writer.writerows(gt_res)

    return 0


def main():
    image_name_list = glob.glob(CROP_IMG_PATH + '*')
    train_set, test_set = train_test_split(image_name_list, test_size=0.51)
    train_part_A, train_part_B = train_test_split(train_set, test_size=0.7)
    test_part_A, test_part_B = train_test_split(test_set, test_size=0.7)

    print('Train set size: ', len(train_set))
    print('Train set size: ', len(test_set))
    print('Train set part A size: ', len(train_part_A))
    print('Train set part B size: ', len(train_part_B))
    print('Test set part A size: ', len(test_part_A))
    print('Test set part B size: ', len(test_part_B))

    generate_dataset(train_part_A, train_part_B)
    generate_dataset(test_part_A, test_part_B, False)


if __name__ == '__main__':
    main()
