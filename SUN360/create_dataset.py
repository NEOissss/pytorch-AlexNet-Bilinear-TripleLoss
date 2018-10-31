import os
import numpy as np
import cv2
import csv
import glob
import random
import pickle
from sklearn.model_selection import train_test_split

ROOT_PATH = '/mnt/nfs/scratch1/gluo/SUN360/'
PATH = ROOT_PATH + 'SUN360_9104x4552/'
CROP_IMG_PATH = ROOT_PATH + 'HalfHalf/IMGs/'
CROP_RATIO = 0.2
RAND_RATIO = 0.8
SIZE = 1400


def crop_img(img_path):
    im = cv2.imread(img_path)
    height, width = im.shape[:2]
    crop_height = int(height * CROP_RATIO)
    #im = im[crop_height:-crop_height, :]
    rand_range = int(((height - 2*crop_height)//2 - SIZE//2) * RAND_RATIO)
    left_x = crop_height + (height - 2*crop_height)//2 - SIZE//2 + random.randint(-rand_range, rand_range)
    left_y = width//2 - width//4 - SIZE//2 + random.randint(-rand_range, rand_range)
    right_x = crop_height + (height - 2*crop_height)//2 - SIZE//2 + random.randint(-rand_range, rand_range)
    right_y = width//2 + width//4 - SIZE//2 + random.randint(-rand_range, rand_range)

    left_crop = im[left_x:left_x+SIZE, left_y:left_y+SIZE, :]
    right_crop = im[right_x:right_x+SIZE, right_y:right_y+SIZE, :]
    return left_crop, right_crop


def get_img_list(part):
    hist_left, hist_right = [], []

    for i in range(max(len(part), 120)):
        im_l, im_r = crop_img(part[i])

        dir_path = CROP_IMG_PATH + part[i].split('/')[-1].split('.')[0]
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        cv2.imwrite(dir_path + '/L.jpg', im_l)
        cv2.imwrite(dir_path + '/R.jpg', im_r)

        print(part[i])
        print('cropped and saved.')

        hist_left.append(cv2.calcHist([im_l], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]))
        hist_right.append(cv2.calcHist([im_r], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]))

    return hist_left, hist_right


def generate_dataset(part_A, part_B, train=True):
    if train:
        task_path = ROOT_PATH + 'HalfHalf/task_train'
        gt_path = ROOT_PATH + 'HalfHalf/gt_train.csv'
        print('Start generate half&half training set...')
    else:
        task_path = ROOT_PATH + 'HalfHalf/task_test'
        gt_path = ROOT_PATH + 'HalfHalf/gt_test.csv'
        print('Start generate half&half testing set...')

    num_A = len(part_A)
    num_B = len(part_B)

    hist_A_left, hist_A_right = get_img_list(part_A)
    hist_B_left, hist_B_right = get_img_list(part_B)

    if train:
        prefix = 'train'
    else:
        prefix = 'test'
    with open(ROOT_PATH+'HalfHalf/'+prefix+'hist_A_left.pickle', 'wb') as f:
        pickle.dump(hist_A_left, f)
    with open(ROOT_PATH+'HalfHalf/'+prefix+'hist_A_right.pickle', 'wb') as f:
        pickle.dump(hist_A_right, f)
    with open(ROOT_PATH+'HalfHalf/'+prefix+'hist_B_left.pickle', 'wb') as f:
        pickle.dump(hist_B_left, f)
    with open(ROOT_PATH+'HalfHalf/'+prefix+'hist_B_right.pickle', 'wb') as f:
        pickle.dump(hist_B_right, f)

    print('Anticipating #sample: ', num_A)

    compare_method = 3 # means that the smaller the compareHist, the similar
    num_choice = 10
    K = 100

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

    samples_num = 16 # take samples_num of samples to show the results
    len_sample_id = 9  # e.g 000000666

    gt_res = []

    if not os.path.exists(task_path):
        os.mkdir(task_path)

    for i in range(max(num_A, samples_num)):
        str_i = '0'*(len_sample_id-len(str(i))) + str(i)
        img_list = ['', ['' for i in range(10)]]

        if task_ref[i] < 0 or (task_ref[i] == 0 and sum(task_choices_minus_1[i])>0): # ref in A_left, ground_truth in A_right, target choices in B_right
            # if not os.path.exists(task_path+'/'+str_i):
            #     os.mkdir(task_path+'/'+str_i)
            crop_img_path = part_A[i].split('/')[-1].split('.')[0] + '/L.jpg'
            img_list[0] = crop_img_path
            # cv2.imwrite(task_path+'/'+str_i+"/reference" + ".jpg", im_A_left[i])

            gt_id = random.randint(0, num_choice-1) # randomly set up gt_id from {0, 1, 2, ...., num_choice-1}
            crop_img_path = part_A[i].split('/')[-1].split('.')[0] + '/R.jpg'
            img_list[1][gt_id] = crop_img_path
            # cv2.imwrite(task_path+'/'+str_i+"/choice_" + str(gt_id) + ".jpg", im_A_right[i])

            gt_res.append([str_i, str(gt_id)])

            for choice_i in range(num_choice):
                if choice_i < gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i]].split('/')[-1].split('.')[0] + '/R.jpg'
                    img_list[1][choice_i] = crop_img_path
                    #cv2.imwrite(task_path+'/'+str_i+"/choice_"+str(choice_i)+ ".jpg", im_B_right[task_choices_minus_1[i][choice_i]])
                elif choice_i > gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i-1]].split('/')[-1].split('.')[0] + '/R.jpg'
                    img_list[1][choice_i] = crop_img_path
                    #cv2.imwrite(task_path+'/'+str_i+"/choice_"+str(choice_i)+ ".jpg", im_B_right[task_choices_minus_1[i][choice_i-1]])


        elif task_ref[i] > 0 or (task_ref[i] == 0 and sum(task_choices_minus_1[i])<0) : # ref in A_right, ground_truth in A_left, target choices in B_left
            # if not os.path.exists(task_path+'/'+str_i):
            #     os.mkdir(task_path+'/'+str_i)
            crop_img_path = part_A[i].split('/')[-1].split('.')[0] + '/R.jpg'
            img_list[0] = crop_img_path
            #cv2.imwrite(task_path+'/'+str_i+"/reference" + ".jpg", im_A_right[i])

            gt_id = random.randint(0, num_choice-1) # randomly set up gt_id from {0, 1, 2, ...., num_choice-1}
            gt_id = random.randint(0, num_choice-1) # randomly set up gt_id from {0, 1, 2, ...., num_choice-1}
            crop_img_path = part_A[i].split('/')[-1].split('.')[0] + '/L.jpg'
            img_list[1][gt_id] = crop_img_path
            #cv2.imwrite(task_path+'/'+str_i+"/choice_" + str(gt_id) + ".jpg", im_A_left[i])

            gt_res.append([str_i, str(gt_id)])

            for choice_i in range(num_choice):
                if choice_i < gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i]*(-1)].split('/')[-1].split('.')[0] + '/L.jpg'
                    img_list[1][choice_i] = crop_img_path
                    #cv2.imwrite(task_path+'/'+str_i+"/choice_"+str(choice_i)+ ".jpg", im_B_left[task_choices_minus_1[i][choice_i] * (-1)])
                elif choice_i > gt_id:
                    crop_img_path = part_B[task_choices_minus_1[i][choice_i-1]*(-1)].split('/')[-1].split('.')[0] + '/L.jpg'
                    img_list[1][choice_i] = crop_img_path
                    #cv2.imwrite(task_path+'/'+str_i+"/choice_"+str(choice_i)+ ".jpg", im_B_left[task_choices_minus_1[i][choice_i-1] * (-1)])

        with open(task_path+'/'+str_i+'.pickle', 'wb') as f:
            pickle.dump(img_list, f)

        if train:
            print("Training example " + str_i + ' generated')
        else:
            print("Testing example " + str_i + ' generated')

    print(len(gt_res), 'samples generated.')

    with open(gt_path,"w+") as my_csv:            # writing the file as my_csv
        csvWriter = csv.writer(my_csv,delimiter=',')  # using the csv module to write the file
        csvWriter.writerows(gt_res)

    return 0


def main():
    image_name_list = glob.glob(PATH + '**/*.jpg*', recursive = True) # the whole image dataset, 118287 images
    train_set, test_set = train_test_split(image_name_list, test_size = 0.51)# train-99,192 images, test-103,242 images.
    train_part_A, train_part_B = train_test_split(train_set, test_size = 0.7)# train_part_A-29,757, train_part_B-69,435 images
    test_part_A, test_part_B = train_test_split(test_set, test_size = 0.7) # test_part_A-30,972, test_part_B-72,270 images

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