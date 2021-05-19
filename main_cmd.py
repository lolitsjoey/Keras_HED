import main_hyperspec_pca
import main_autodct_analysis
import main_autofft_analysis
import main_edge_detector
import pandas as pd
import os
import tensorflow as tf
#train

if __name__ == '__main__':
    #with tf.device('CPU:0'):
        folder_index = ['100-cft-big', '100-gen-big', '100-gen', 'A13', 'A16', 'A20-2', 'C26522']
        num_in_each = [190, 251, 80, 13, 16, 20, 3]

        index_list = []
        for fd, num in zip(folder_index, num_in_each):
            index_list = index_list + [fd + '_' + str(i) for i in range(num)]
        scored_frame = pd.DataFrame(
                    index=['_'.join(note.split('_')[0:2]) for note in index_list])
        for folder in os.listdir('D:/scoring_and_profiling/'):
            if ('edges' in folder) or ('dct' in folder) or ('fft' in folder) or ('cwt' in folder):
                continue
            folder_to_score = 'D:/scoring_and_profiling/' + folder + '/'
            print('-------' + folder_to_score + '--------')
            #scored_frame = main_hyperspec_pca.main(folder_to_score, scored_frame)
            scored_frame = main_edge_detector.main(folder_to_score, scored_frame)
            scored_frame = main_autodct_analysis.main(folder_to_score, scored_frame)
            scored_frame = main_autofft_analysis.main(folder_to_score, scored_frame)


            scored_frame.to_csv('./' + folder + '_big_data_no_doss.csv')
#main_autofft_analysis.main(folder_to_score)
