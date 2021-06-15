import main_hyperspec_pca
import main_autodct_analysis
import main_hyperspec_analysis
import main_autofft_analysis
import main_edge_detector
import pandas as pd
import os
import tensorflow as tf
import math
#train

if __name__ == '__main__':
    folder_index = ['C26459', 'C26522', 'A20-2', 'A16', 'A13', '100 - new']
    num_in_each = [12, 12, 80, 64, 52, 320]

    index_list = []
    for fd, num in zip(folder_index, num_in_each):
        index_list = index_list + [fd + '_' + str(math.floor(i / 4)) + '_' + str(i % 4) for i in range(num)]
    scored_frame = pd.DataFrame(
        index=index_list)
    for folder in os.listdir('D:/scoring_and_profiling/')[29::]:
        if ('edges' in folder) or ('dct' in folder) or ('fft' in folder) or ('cwt' in folder) or ('waves' in folder) or ('aug' in folder):
            continue
        folder_to_score = 'D:/scoring_and_profiling/' + folder + '/'
        print('-------' + folder_to_score + '--------')

        scores_df = main_hyperspec_analysis.main(folder_to_score, scored_frame, 0)
        # scored_frame = main_autodct_analysis.main(folder_to_score, scored_frame)
        # scored_frame = main_autofft_analysis.main(folder_to_score, scored_frame)
        # scored_frame = main_edge_detector.main(folder_to_score, scored_frame)

        scored_frame.to_csv('./' + folder + '.csv')
#main_autofft_analysis.main(folder_to_score)
