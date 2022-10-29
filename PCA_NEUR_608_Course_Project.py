

## LOAD MODULES/SOFTWARE
import os
import glob
import pandas as pd
import numpy as np

import sys
import pickle
import scipy
from scipy.io import savemat, loadmat
from scipy import stats
import covbat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedFormatter, FixedLocator
from numpy import genfromtxt

###################################################################################################################
##################                                INPUT MATRIX                                #####################
##################              CODE ADAPTED FROM https://github.com/CoBrALab/cobra-nmf       #####################
###################################################################################################################

#Read in csv with subject demographics 
df_sorted = pd.read_csv('C:\\Users\\fabbo\\Desktop\\Course_Project\\Phenotypic_V2_input.csv') #MODIFY to replace the .csv filename with the path to your sorted demographics csv file


n_subjects=df_sorted.shape[0] #num rows in spreadsheet
n_vertex=40962


#Load left thicknesses into matrix left_ct
#do first row to create matrix, concat from there
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\thickness_native_rms_rsl_tlink_30mm_left\\" + str(row) + '_native_rms_rsl_tlink_30mm_left.txt' 
left_ct = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(left_ct)) #1 x 40962

#load thickness file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\thickness_native_rms_rsl_tlink_30mm_left\\" + str(row) + '_native_rms_rsl_tlink_30mm_left.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    left_ct = np.concatenate((left_ct, x), axis=0)
print("raw left has", np.shape(left_ct)[0], "subjects", np.shape(left_ct)[1], "vertices")

#Repeat for right side
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\thickness_native_rms_rsl_tlink_30mm_right\\" + str(row) + '_native_rms_rsl_tlink_30mm_right.txt'
right_ct = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(right_ct)) #1 x 40962
#load thickness file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\thickness_native_rms_rsl_tlink_30mm_right\\" + str(row) + '_native_rms_rsl_tlink_30mm_right.txt'
    x = np.loadtxt(fname).reshape(1,n_vertex)
    right_ct = np.concatenate((right_ct, x), axis=0)
print("raw right has", np.shape(right_ct)[0], "subjects", np.shape(right_ct)[1], "vertices")


#write out in .mat format
#PCA input - subject x vertex
out_matrix = np.asmatrix(left_ct)
fname = "left_ct_raw.mat"
print("Saving raw left ct to", fname,  "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

out_matrix = np.asmatrix(right_ct)
fname = "right_ct_raw.mat"
print("Saving raw right ct to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix


#Concatenate the left and right hemisphere data to get whole brain data
wb_ct_1= np.concatenate((np.transpose(left_ct), np.transpose(right_ct)),axis=0)
wb_ct = np.transpose(wb_ct_1)

#write out whole brain data in .mat format
out_matrix = np.asmatrix(wb_ct)
fname = "wholebrain_ct_raw.mat"
print("Saving whole brain ct to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})


#Load left volumes into matrix left_vol
#do first row to create matrix, concat from there
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_surface_rsl_left_native_volume_40mm\\" + str(row) + '_surface_rsl_left_native_volume_40mm.txt' 
left_vol = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(left_vol)) #1 x 40962

#load volume file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_surface_rsl_left_native_volume_40mm\\" + str(row) + '_surface_rsl_left_native_volume_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    left_vol = np.concatenate((left_vol, x), axis=0)
print("raw left has", np.shape(left_vol)[0], "subjects", np.shape(left_vol)[1], "vertices")
#Repeat for right side
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_surface_rsl_right_native_volume_40mm\\" + str(row) + '_surface_rsl_right_native_volume_40mm.txt' 
right_vol = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(right_vol)) #1 x 40962
#load volume file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_surface_rsl_right_native_volume_40mm\\" + str(row) + '_surface_rsl_right_native_volume_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    right_vol = np.concatenate((right_vol, x), axis=0)
print("raw right has", np.shape(right_vol)[0], "subjects", np.shape(right_vol)[1], "vertices")


#write out in .mat format
#PCA input - subject x vertex
out_matrix = np.transpose(np.asmatrix(left_vol))
fname = "left_vol_raw.mat"
print("Saving raw left vol to", fname,  "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

out_matrix = np.transpose(np.asmatrix(right_vol))
fname = "right_vol_raw.mat"
print("Saving raw right vol to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

#Concatenate the left and right hemisphere data to get whole brain data
wb_vol_1 = np.concatenate((np.transpose(left_vol), np.transpose(right_vol)),axis=0)
wb_vol = np.transpose(wb_vol_1)

#write out whole brain data in .mat format
out_matrix = np.asmatrix(wb_vol)
fname = "wholebrain_vol_raw.mat"
print("Saving whole brain vol to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})


#Load left surface area into matrix left_sa
#do first row to create matrix, concat from there
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_mid_surface_rsl_left_native_area_40mm\\" + str(row) + '_mid_surface_rsl_left_native_area_40mm.txt' 
left_sa = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(left_sa)) #1 x 40962

#load volume file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_mid_surface_rsl_left_native_area_40mm\\" + str(row) + '_mid_surface_rsl_left_native_area_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    left_sa = np.concatenate((left_sa, x), axis=0)
print("raw left has", np.shape(left_sa)[0], "subjects", np.shape(left_sa)[1], "vertices")

#Repeat for right side
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_mid_surface_rsl_right_native_area_40mm\\" + str(row) + '_mid_surface_rsl_right_native_area_40mm.txt' 
right_sa = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(right_sa)) #1 x 40962
#load surface area file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset_No_Controls\\surfaces_mid_surface_rsl_right_native_area_40mm\\" + str(row) + '_mid_surface_rsl_right_native_area_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    right_sa = np.concatenate((right_sa, x), axis=0)
print("raw right has", np.shape(right_sa)[0], "subjects", np.shape(right_sa)[1], "vertices")


#write out in .mat format
#PCA input - subject x vertex
out_matrix = np.transpose(np.asmatrix(left_sa))
fname = "left_vol_raw.mat"
print("Saving raw left sa to", fname,  "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

out_matrix = np.transpose(np.asmatrix(right_sa))
fname = "right_vol_raw.mat"
print("Saving raw right sa to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

#Concatenate the left and right hemisphere data to get whole brain data
wb_sa_1 = np.concatenate((np.transpose(left_sa), np.transpose(right_sa)),axis=0)
wb_sa = np.transpose(wb_sa_1)

#write out whole brain data in .mat format
out_matrix = np.asmatrix(wb_sa)
fname = "wholebrain_sa_raw.mat"
print("Saving whole brain sa to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})


#standardize matrices
input_list=["wholebrain_ct_raw","wholebrain_vol_raw","wholebrain_sa_raw"] #name of the file without the .mat extension

z_dict = {}

for metric in input_list:
    fname = metric + ".mat"
    res = loadmat(fname) #load raw data
    x_z = np.asarray(stats.zscore(res['X'],axis=None)) #zscore, across both subjects and vertices
    z_dict[metric] = x_z

#concatenate each zscored shifted matrix together 
#forms vertex X subject*n_metrics matrix
metric=input_list[0]
wb_z_all = z_dict[metric]
#if only one metric, skip the for loop below, go straight to line 38 #
for metric in input_list[1:]:
    print(metric)
    wb_z_all = np.concatenate((wb_z_all, z_dict[metric]),axis=1)

#write out z scored, shifted data for whole group analysis
fname = sys.argv[1]
print(fname, np.shape(wb_z_all))
savemat(fname, {'X': wb_z_all})


#heat mapping for input matrix
def heatmapping(data,minn,maxx,cbar_tix,fig_width,fig_height,title='',fname=''):
    import matplotlib as mpl
    from matplotlib import cm
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedFormatter, FixedLocator
    plt.rc('figure', titlesize=30)  # fontsize of the figure title
    #Linearly interpoalte a colour gradient 
   
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    cmap = mpl.colors.ListedColormap(newcolors)
    img = plt.imshow(data,interpolation='nearest',     cmap = cmap, origin='upper',vmin=minn,vmax=maxx)
    #Set the axis of the plot so it isn't a long rectangle
    ax = plt.gca()
    ax.set_aspect('auto') 
    ax.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='on',left='on',labelleft='on', pad = 5)
    ax.set_xlabel('')
    ax.set_ylabel('Subjects', fontsize=30)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.labelpad = 5
    ax.tick_params(axis='y',size=15)
    ax.grid(False)
    fig = plt.gcf()
    fig.set_size_inches(fig_width,fig_height)
    n_metrics = 5
    #n_subj = np.shape(data)[1]/5
    #for x in range(1,n_metrics):
    #    plt.axvline(x=(n_subj*x),c='w',linewidth=2)
    #Generate a color bar
    cbar = plt.colorbar(img,cmap=cmap)
    
    cbar.set_ticks(np.arange(minn, maxx, cbar_tix))
    cbar.ax.tick_params(labelsize=12)
    if title:
        plt.title(title, fontsize=30)
    plt.savefig('inputmatrix.png', bbox_inches='tight')
    

heatmapping(wb_z_all,np.min(wb_z_all),np.max(wb_z_all),0.5,16,8,title="Input",fname=sys.argv[2]) 

#====================================================================================================================================================

###################################################################################################################
######################                                PCA                                 #########################
###################################################################################################################

#apply PCA to matrix
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(wb_z_all)


#assess variance explained by each component
print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)


#create scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='black')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#====================================================================================================================================================

###################################################################################################################
################                            HARMONIZED DATA ANALYSIS                           ####################
###################################################################################################################

#harmonize data using covbat
wb_cov_ct1 = np.transpose(wb_ct)
wb_cov_ct = pd.DataFrame(wb_cov_ct1)

wb_cov_vol1 = np.transpose(wb_vol)
wb_cov_vol = pd.DataFrame(wb_cov_vol1)

wb_cov_sa1 = np.transpose(wb_sa)
wb_cov_sa = pd.DataFrame(wb_cov_sa1)



#harmonize data using covbat
batch1 = genfromtxt('C:\\Users\\fabbo\\Desktop\\Course_Project\\batch.csv', delimiter=',')
batch = pd.Series(batch1)

wb_ct_har = covbat.covbat(data=wb_cov_ct, batch=batch)
wb_vol_har = covbat.covbat(data=wb_cov_vol, batch=batch)
wb_sa_har = covbat.covbat(data=wb_cov_sa, batch=batch)


#write out harmonized matrices in .mat format
out_matrix = np.transpose(np.asmatrix(wb_ct_har))
fname = "wholebrain_ct_har.mat"
print("Saving harmonized ct to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})

out_matrix = np.transpose(np.asmatrix(wb_vol_har))
fname = "wholebrain_vol_har.mat"
print("Saving harmonized vol to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})

out_matrix = np.transpose(np.asmatrix(wb_sa_har))
fname = "wholebrain_sa_har.mat"
print("Saving harmonized sa to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})

#standardize matrices
input_list=["wholebrain_ct_har","wholebrain_vol_har","wholebrain_sa_har"] #name of the file without the .mat extension

z_dict = {}

for metric in input_list:
    fname = metric + ".mat"
    res = loadmat(fname) #load raw data
    x_z = np.asarray(stats.zscore(res['X'],axis=None)) #zscore, across both subjects and vertices
    z_dict[metric] = x_z


#concatenate each zscored shifted matrix together 
#forms vertex X subject*n_metrics matrix
metric=input_list[0]
wb_z_har = z_dict[metric]
#if only one metric, skip the for loop below, go straight to line 38 #
for metric in input_list[1:]:
    print(metric)
    wb_z_all = np.concatenate((wb_z_har, z_dict[metric]),axis=1)

#write out z scored, shifted data for whole group analysis
fname = sys.argv[1]
print(fname, np.shape(wb_z_har))
savemat(fname, {'X': wb_z_har})


plt.switch_backend('Agg')
fname=sys.argv[1]
x=scipy.io.loadmat(fname)['X']
print(np.shape(x))
print(np.min(x), np.mean(x), np.max(x))

#apply PCA to harmonized matrix
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(wb_z_all)


#assess variance explained by each component
print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)


#create scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='black')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

