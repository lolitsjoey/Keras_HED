import main_hyperspec_pca
import main_autodct_analysis
import main_autofft_analysis
import main_edge_detector
import pandas as pd
import os
import tensorflow as tf
#train

if __name__ == '__main__':
    with tf.device('CPU:0'):
        scored_frame = pd.DataFrame(
                    index=['_'.join(note.split('_')[0:2]) for note in os.listdir('D:/scoring_and_profiling/FedSeal/')])
        for folder in os.listdir('D:/scoring_and_profiling/'):
            if ('edges' in folder) or ('dct' in folder) or ('fft' in folder) or ('cwt' in folder):
                continue
            folder_to_score = 'D:/scoring_and_profiling/' + folder + '/'
            print('-------' + folder_to_score + '--------')
            scored_frame = main_hyperspec_pca.main(folder_to_score, scored_frame)
            scored_frame = main_edge_detector.main(folder_to_score, scored_frame)
            scored_frame = main_autodct_analysis.main(folder_to_score, scored_frame)
            scored_frame = main_autofft_analysis.main(folder_to_score, scored_frame)


            scored_frame.to_csv('./' + folder + '_cmdovernight.csv')
#main_autofft_analysis.main(folder_to_score)
