import os
import pandas as pd
import shutil

# 1/250 sample rate for each slide; exclude 76/79/80
if __name__ == "__main__":
    newpd = pd.DataFrame(columns=['Num','X_pos','Y_pos','X','Y','X_relative','Y_relative','tile_file'])
    ignore = ['.DS_Store', 'dict.csv', 'rename.s']
    for id in os.listdir('../tiles'):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            dirname = id
            exclude = ['0000026276', '0000026279', '0000026280']
            dic = pd.read_csv('../tiles/{}/{}_dict.csv'.format(dirname, dirname), header=0)
            if dirname in exclude:
                print('Skipping slide:', dirname)
            else:
                sampled_dic = dic.sample(int(dic.shape[0]/250), replace=False)
                newpd=pd.concat([newpd, sampled_dic])
    newpd.to_csv('../sampled_tiles/sampled_for_label_batch2.csv', index=False)

    samplelist = pd.read_csv('../sampled_tiles/sampled_for_label_batch2.csv', header=0)
    for idx, row in samplelist.iterrows():
        shutil.copyfile(str(row['tile_file']), '../sampled_tiles/{}'.format(str(row['tile_file']).split('/')[-1]))



