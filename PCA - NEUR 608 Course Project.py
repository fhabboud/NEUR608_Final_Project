#!/usr/bin/env python
# coding: utf-8

# In[1]:
#LOAD MODULES and SOFTWARE

import os
import glob
import pandas as pd
import numpy as np

import sys
import pickle
import scipy
import pyls
from scipy.io import savemat, loadmat
from scipy import stats
import covbat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedFormatter, FixedLocator
from numpy import genfromtxt
from pyls import behavioral_pls

# In[2]:


#Read in csv with subject demographics 
df_sorted = pd.read_csv('C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Beh_Input.csv') #MODIFY to replace the .csv filename with the path to your sorted demographics csv file
left_mask = np.loadtxt('C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608\\CIVET_2.0_mask_left_short.txt') #MODIFY to point to your left mask file
left_valid = np.where(left_mask==1) # list of valid indices in civet .txt file
left_invalid = np.where(left_mask==0) #list of invalid indices in civet .txt file

right_mask = np.loadtxt('C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608\\CIVET_2.0_mask_right_short.txt') #MODIFY to point to your right mask file
right_valid = (np.where(right_mask==1))
right_invalid = (np.where(right_mask==0))

# In[3]:


n_subjects=df_sorted.shape[0] #num rows in spreadsheet
n_vertex=40962


# In[4]:


#Load left thicknesses into matrix left_ct
#do first row to create matrix, concat from there
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\thickness_native_rms_rsl_tlink_30mm_left\\" + str(row) + '_native_rms_rsl_tlink_30mm_left.txt' 
left_ct = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(left_ct)) #1 x 40962

#load thickness file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\thickness_native_rms_rsl_tlink_30mm_left\\" + str(row) + '_native_rms_rsl_tlink_30mm_left.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    left_ct = np.concatenate((left_ct, x), axis=0)
print("raw left has", np.shape(left_ct)[0], "subjects", np.shape(left_ct)[1], "vertices")

#Repeat for right side
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\thickness_native_rms_rsl_tlink_30mm_right\\" + str(row) + '_native_rms_rsl_tlink_30mm_right.txt'
right_ct = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(right_ct)) #1 x 40962
#load thickness file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\thickness_native_rms_rsl_tlink_30mm_right\\" + str(row) + '_native_rms_rsl_tlink_30mm_right.txt'
    x = np.loadtxt(fname).reshape(1,n_vertex)
    right_ct = np.concatenate((right_ct, x), axis=0)
print("raw right has", np.shape(right_ct)[0], "subjects", np.shape(right_ct)[1], "vertices")

#%%
#valid ct matrices have dimensions subj x validvertices, masked midline vertices removed
ct_rows = np.shape(left_ct)[0] #same for right side
ct_cols = np.shape(left_valid)[1] #same for right side

#take all rows (subjects), but only columns which are valid vertices
left_ct_valid = left_ct[:, left_valid].reshape(ct_rows,ct_cols)
right_ct_valid = right_ct[:, right_valid].reshape(ct_rows,ct_cols)

#write out in .mat format
out_matrix = np.asmatrix(left_ct_valid)
fname = "left_ct_raw.mat"
print("Saving raw left ct to", fname,  "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

out_matrix = np.asmatrix(right_ct_valid)
fname = "right_ct_raw.mat"
print("Saving raw right ct to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

# In[6]:


#Concatenate the left and right hemisphere data to get whole brain data
wb_ct = np.concatenate((left_ct_valid, right_ct_valid),axis=1)

#write out whole brain data in .mat format
out_matrix = np.asmatrix(wb_ct)
fname = "wholebrain_ct_raw.mat"
print("Saving whole brain ct to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})


# In[7]:


#Load left volumes into matrix left_vol
#do first row to create matrix, concat from there
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_surface_rsl_left_native_volume_40mm\\" + str(row) + '_surface_rsl_left_native_volume_40mm.txt' 
left_vol = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(left_vol)) #1 x 40962

#load volume file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_surface_rsl_left_native_volume_40mm\\" + str(row) + '_surface_rsl_left_native_volume_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    left_vol = np.concatenate((left_vol, x), axis=0)
print("raw left has", np.shape(left_vol)[0], "subjects", np.shape(left_vol)[1], "vertices")
#Repeat for right side
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_surface_rsl_right_native_volume_40mm\\" + str(row) + '_surface_rsl_right_native_volume_40mm.txt' 
right_vol = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(right_vol)) #1 x 40962
#load volume file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_surface_rsl_right_native_volume_40mm\\" + str(row) + '_surface_rsl_right_native_volume_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    right_vol = np.concatenate((right_vol, x), axis=0)
print("raw right has", np.shape(right_vol)[0], "subjects", np.shape(right_vol)[1], "vertices")


#%%
#valid vol matrices have dimensions subj x validvertices, masked midline vertices removed
vol_rows = np.shape(left_vol)[0] #same for right side
vol_cols = np.shape(left_valid)[1] #same for right side

#take all rows (subjects), but only columns which are valid vertices
left_vol_valid = left_vol[:, left_valid].reshape(vol_rows,vol_cols)
right_vol_valid = right_vol[:, right_valid].reshape(vol_rows,vol_cols)

#write out in .mat format
out_matrix = np.transpose(np.asmatrix(left_vol_valid))
fname = "left_vol_raw.mat"
print("Saving raw left vol to", fname,  "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

out_matrix = np.transpose(np.asmatrix(right_vol_valid))
fname = "right_vol_raw.mat"
print("Saving raw right vol to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix


# In[9]:


#Concatenate the left and right hemisphere data to get whole brain data
wb_vol = np.concatenate((left_vol_valid, right_vol_valid),axis=1)

#write out whole brain data in .mat format
out_matrix = np.asmatrix(wb_vol)
fname = "wholebrain_vol_raw.mat"
print("Saving whole brain vol to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})


# In[10]:


#Load left surface area into matrix left_sa
#do first row to create matrix, concat from there
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_mid_surface_rsl_left_native_area_40mm\\" + str(row) + '_mid_surface_rsl_left_native_area_40mm.txt' 
left_sa = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(left_sa)) #1 x 40962

#load volume file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_mid_surface_rsl_left_native_area_40mm\\" + str(row) + '_mid_surface_rsl_left_native_area_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    left_sa = np.concatenate((left_sa, x), axis=0)
print("raw left has", np.shape(left_sa)[0], "subjects", np.shape(left_sa)[1], "vertices")

#Repeat for right side
row = df_sorted['FILE_ID'].tolist()[0] 
fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_mid_surface_rsl_right_native_area_40mm\\" + str(row) + '_mid_surface_rsl_right_native_area_40mm.txt' 
right_sa = np.loadtxt(fname).reshape(1,n_vertex)
print(np.shape(right_sa)) #1 x 40962
#load surface area file for rest of subjects and concatenate to make subjects X vertices matrix
for row in df_sorted['FILE_ID'].tolist()[1:]: 
    fname="C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Dataset\\surfaces_mid_surface_rsl_right_native_area_40mm\\" + str(row) + '_mid_surface_rsl_right_native_area_40mm.txt' 
    x = np.loadtxt(fname).reshape(1,n_vertex)
    right_sa = np.concatenate((right_sa, x), axis=0)
print("raw right has", np.shape(right_sa)[0], "subjects", np.shape(right_sa)[1], "vertices")


#%%
#valid sa matrices have dimensions subj x validvertices, masked midline vertices removed
sa_rows = np.shape(left_sa)[0] #same for right side
sa_cols = np.shape(left_valid)[1] #same for right side

#take all rows (subjects), but only columns which are valid vertices
left_sa_valid = left_sa[:, left_valid].reshape(sa_rows,sa_cols)
right_sa_valid = right_sa[:, right_valid].reshape(sa_rows,sa_cols)

#write out in .mat format

out_matrix = np.transpose(np.asmatrix(left_sa_valid))
fname = "left_vol_raw.mat"
print("Saving raw left sa to", fname,  "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix

out_matrix = np.transpose(np.asmatrix(right_sa_valid))
fname = "right_vol_raw.mat"
print("Saving raw right sa to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})
del out_matrix


# In[12]:


#Concatenate the left and right hemisphere data to get whole brain data
wb_sa = np.concatenate((left_sa_valid, right_sa_valid),axis=1)

#write out whole brain data in .mat format
out_matrix = np.asmatrix(wb_sa)
fname = "wholebrain_sa_raw.mat"
print("Saving whole brain sa to", fname, "with shape", np.shape(out_matrix))
scipy.io.savemat(fname, {'X': out_matrix})


# In[13]:


#standardize matrices
input_list=["wholebrain_ct_raw","wholebrain_vol_raw","wholebrain_sa_raw"] #name of the file without the .mat extension

z_dict = {}

for metric in input_list:
    fname = metric + ".mat"
    res = loadmat(fname) #load raw data
    x_z = np.asarray(stats.zscore(res['X'],axis=None)) #zscore, across both subjects and vertices
    z_dict[metric] = x_z


# In[14]:


#concatenate each zscored shifted matrix together 
#forms vertex X subject*n_metrics matrix
metric=input_list[0]
wb_z_all = z_dict[metric]
for metric in input_list[1:]:
    print(metric)
    wb_z_all = np.concatenate((wb_z_all, z_dict[metric]),axis=1)


# In[15]:


#write out z scored, shifted data for whole group analysis
fname = sys.argv[1]
print(fname, np.shape(wb_z_all))
savemat(fname, {'X': wb_z_all})


# In[16]:


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


# In[17]:


#apply PCA to matrix
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(wb_z_all)


# In[18]:


#assess variance explained by each component
print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)


# In[19]:


#create scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='black')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.savefig('PCA.svg',bbox_inches='tight', transparent=True)
plt.show()

# In[21]:

#harmonize data using covbat
wb_z_har_1 = pd.DataFrame(np.transpose(wb_z_all))
batch1 = genfromtxt('C:\\Users\\fabbo\\Desktop\\Course_Project\\batch.csv', delimiter=',')
batch = pd.Series(batch1)

wb_z_har_2 = covbat.covbat(data=wb_z_har_1, batch=batch)


# In[27]:


#apply PCA to harmonized matrix
wb_z_har_3 = np.array(wb_z_har_2)
wb_z_har = np.transpose(wb_z_har_3)
pca_har = PCA(n_components=20)
principalComponents_har = pca_har.fit_transform(wb_z_har)


# In[28]:


#assess variance explained by each component
print(pca_har.explained_variance_ratio_)
sum(pca_har.explained_variance_ratio_)


# In[29]:


#create scree plot
PC_values = np.arange(pca_har.n_components_) + 1
plt.plot(PC_values, pca_har.explained_variance_ratio_, 'o-', linewidth=2, color='black')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.savefig('PCA_Har.svg',bbox_inches='tight', transparent=True)
plt.show()

# In[30]:

#run three component PCA only
pca_2 = PCA(n_components=2)
principalComponents_2 = pca_2.fit_transform(wb_z_all)


#%%
loadings = pd.DataFrame(pca_2.components_.T, columns=['PC1', 'PC2'])
np.savetxt("PCA_Loadings.csv", loadings, delimiter=",")

#%%

#load components
PC1 = np.array(loadings.iloc[:, 0])
PC2 = np.array(loadings.iloc[:, 1])

#%%
#separate components based on metric
PC1_ct = (PC1[0:77122,])*1000
PC1_vol = (PC1[77122:154244,])*1000
PC1_sa = (PC1[154244:231366,])*1000
PC2_ct = (PC2[0:77122,])*1000
PC2_vol = (PC2[77122:154244,])*1000
PC2_sa = (PC2[154244:231366,])*1000

#%%
n_vertex = 77122
PC1_ct = np.transpose(PC1_ct.reshape(1,n_vertex))
PC1_vol = np.transpose(PC1_vol.reshape(1,n_vertex))
PC1_sa = np.transpose(PC1_sa.reshape(1,n_vertex))
PC2_ct = np.transpose(PC2_ct.reshape(1,n_vertex))
PC2_vol = np.transpose(PC2_vol.reshape(1,n_vertex))
PC2_sa = np.transpose(PC2_sa.reshape(1,n_vertex))
#%%
#grab loading for each hemisphere for each PC-metric
#PC1-ct
out_dir = 'C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608'
compnum=np.shape(PC1_ct)[1]
left_PC1_ct=PC1_ct[0:38561,:]
right_PC1_ct=PC1_ct[38561:,:]
left_outarray = np.zeros((np.shape(left_mask)[0],compnum+1))

for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(left_mask)):
        if left_mask[idx] == 1:
            left_outarray[idx,comp] = left_PC1_ct[valid_idx,comp]
            valid_idx +=1
left_statmap = out_dir + "/left_PC1_ct" + str(compnum) + ".txt"
np.savetxt(left_statmap,left_outarray, fmt='%f')

right_outarray = np.zeros((np.shape(right_mask)[0],compnum+1))
for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(right_mask)):
        if right_mask[idx] == 1:
            right_outarray[idx,comp] = right_PC1_ct[valid_idx,comp]
            valid_idx +=1
right_statmap = out_dir + "/right_PC1_ct" + str(compnum) + ".txt"
np.savetxt(right_statmap,right_outarray, fmt='%f')
#%%
#PC1-vol
out_dir = 'C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608'
compnum=np.shape(PC1_vol)[1]
left_PC1_vol=PC1_vol[0:38561,:]
right_PC1_vol=PC1_vol[38561:,:]

left_outarray = np.zeros((np.shape(left_mask)[0],compnum+1))

for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(left_mask)):
        if left_mask[idx] == 1:
            left_outarray[idx,comp] = left_PC1_vol[valid_idx,comp]
            valid_idx +=1
left_statmap = out_dir + "/left_PC1_vol" + str(compnum) + ".txt"
np.savetxt(left_statmap,left_outarray, fmt='%f')

right_outarray = np.zeros((np.shape(right_mask)[0],compnum+1))
for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(right_mask)):
        if right_mask[idx] == 1:
            right_outarray[idx,comp] = right_PC1_vol[valid_idx,comp]
            valid_idx +=1
right_statmap = out_dir + "/right_PC1_vol" + str(compnum) + ".txt"
np.savetxt(right_statmap,right_outarray, fmt='%f')

#%%
#PC1-sa
out_dir = 'C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608'
compnum=np.shape(PC1_sa)[1]
left_PC1_sa=PC1_sa[0:38561,:]
right_PC1_sa=PC1_sa[38561:,:]

left_outarray = np.zeros((np.shape(left_mask)[0],compnum+1))

for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(left_mask)):
        if left_mask[idx] == 1:
            left_outarray[idx,comp] = left_PC1_sa[valid_idx,comp]
            valid_idx +=1
left_statmap = out_dir + "/left_PC1_sa" + str(compnum) + ".txt"
np.savetxt(left_statmap,left_outarray, fmt='%f')

right_outarray = np.zeros((np.shape(right_mask)[0],compnum+1))
for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(right_mask)):
        if right_mask[idx] == 1:
            right_outarray[idx,comp] = right_PC1_sa[valid_idx,comp]
            valid_idx +=1
right_statmap = out_dir + "/right_PC1_sa" + str(compnum) + ".txt"
np.savetxt(right_statmap,right_outarray, fmt='%f')

#%%
#PC2-ct
out_dir = 'C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608'
compnum=np.shape(PC2_ct)[1]
left_PC2_ct=PC2_ct[0:38561,:]
right_PC2_ct=PC2_ct[38561:,:]

left_outarray = np.zeros((np.shape(left_mask)[0],compnum+1))

for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(left_mask)):
        if left_mask[idx] == 1:
            left_outarray[idx,comp] = left_PC2_ct[valid_idx,comp]
            valid_idx +=1
left_statmap = out_dir + "/left_PC2_ct" + str(compnum) + ".txt"
np.savetxt(left_statmap,left_outarray, fmt='%f')

right_outarray = np.zeros((np.shape(right_mask)[0],compnum+1))
for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(right_mask)):
        if right_mask[idx] == 1:
            right_outarray[idx,comp] = right_PC2_ct[valid_idx,comp]
            valid_idx +=1
right_statmap = out_dir + "/right_PC2_ct" + str(compnum) + ".txt"
np.savetxt(right_statmap,right_outarray, fmt='%f')
#%%
#PC2-vol
out_dir = 'C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608'
compnum=np.shape(PC2_vol)[1]
left_PC2_vol=PC2_vol[0:38561,:]
right_PC2_vol=PC2_vol[38561:,:]

left_outarray = np.zeros((np.shape(left_mask)[0],compnum+1))

for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(left_mask)):
        if left_mask[idx] == 1:
            left_outarray[idx,comp] = left_PC2_vol[valid_idx,comp]
            valid_idx +=1
left_statmap = out_dir + "/left_PC2_vol" + str(compnum) + ".txt"
np.savetxt(left_statmap,left_outarray, fmt='%f')

right_outarray = np.zeros((np.shape(right_mask)[0],compnum+1))
for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(right_mask)):
        if right_mask[idx] == 1:
            right_outarray[idx,comp] = right_PC2_vol[valid_idx,comp]
            valid_idx +=1
right_statmap = out_dir + "/right_PC2_vol" + str(compnum) + ".txt"
np.savetxt(right_statmap,right_outarray, fmt='%f')

#%%
#PC2-sa
out_dir = 'C:\\Users\\fabbo\\OneDrive - McGill University\\Courses\\NEUR608'
compnum=np.shape(PC2_sa)[1]
left_PC2_sa=PC2_sa[0:38561,:]
right_PC2_sa=PC2_sa[38561:,:]

left_outarray = np.zeros((np.shape(left_mask)[0],compnum+1))

for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(left_mask)):
        if left_mask[idx] == 1:
            left_outarray[idx,comp] = left_PC2_sa[valid_idx,comp]
            valid_idx +=1
left_statmap = out_dir + "/left_PC2_sa" + str(compnum) + ".txt"
np.savetxt(left_statmap,left_outarray, fmt='%f')

right_outarray = np.zeros((np.shape(right_mask)[0],compnum+1))
for comp in range(0,compnum):
    valid_idx = 0
    for idx in range(0,np.size(right_mask)):
        if right_mask[idx] == 1:
            right_outarray[idx,comp] = right_PC2_sa[valid_idx,comp]
            valid_idx +=1
right_statmap = out_dir + "/right_PC2_sa" + str(compnum) + ".txt"
np.savetxt(right_statmap,right_outarray, fmt='%f')

# In[31]:

beh_var = np.loadtxt('C:\\Users\\fabbo\\Desktop\\Course_Project\\Final_Beh_Input.txt')
print(np.shape(beh_var))


# In[37]:
#run behavioural pls
bpls = behavioral_pls(principalComponents_2, beh_var)
bpls


# In[38]:


help(bpls)


# In[43]:

print(bpls.x_scores)
np.savetxt("LVx.csv", bpls.x_scores, delimiter=",")

#%%
np.shape(bpls.x_scores)

# In[ ]:

print(bpls.y_scores)

#%%
print(bpls.y_weights)

#%%
print(bpls.y_loadings)
np.savetxt("LVy.csv", bpls.y_loadings, delimiter=",")

#%%
#After doing PLS and gotten the number of LCs, checked for permutation significance:

bpls['permres']['pvals']
PLS_p_values = bpls['permres']['pvals']
PLS_p_values_for_correction = PLS_p_values[0:6] #Based on choosing 5 LCs

#%%
import statsmodels.stats.multitest
from statsmodels.stats.multitest import fdrcorrection
fdr = fdrcorrection(PLS_p_values_for_correction, alpha=0.05, method='indep', is_sorted=False)
print(fdr)
#%%
sd1 = np.std(bpls.bootres.y_loadings_boot[0][0])
sd2 = np.std(bpls.bootres.y_loadings_boot[1][0])
sd3 = np.std(bpls.bootres.y_loadings_boot[2][0])
sd4 = np.std(bpls.bootres.y_loadings_boot[3][0])
sd5 = np.std(bpls.bootres.y_loadings_boot[4][0])
sd6 = np.std(bpls.bootres.y_loadings_boot[5][0])
sd7 = np.std(bpls.bootres.y_loadings_boot[6][0])
sd8 = np.std(bpls.bootres.y_loadings_boot[7][0])
sd9 = np.std(bpls.bootres.y_loadings_boot[8][0])
sd10 = np.std(bpls.bootres.y_loadings_boot[9][0])
sd11 = np.std(bpls.bootres.y_loadings_boot[10][0])
sd12 = np.std(bpls.bootres.y_loadings_boot[11][0])

print(sd1)
print(sd2)
print(sd3)
print(sd4)
print(sd5)
print(sd6)
print(sd7)
print(sd8)
print(sd9)
print(sd10)
print(sd11)
print(sd12)
#%%
LVy1 = pd.read_csv(r"LVy1.csv")
LVy1.head()
df_LVy1 = pd.DataFrame(LVy1)

error = [sd1, sd2, sd3, sd4, sd5, sd6, sd7, sd8, sd9, sd10, sd11, sd12]

measure = df_LVy1['Measure'].head(12)
cov = df_LVy1['Cov'].head(12)

fig, ax = plt.subplots(figsize =(20, 12))
ax.barh(measure, cov, xerr = error, color= 'lightcoral')
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
ax.invert_yaxis()
ax.set_xlabel("Correlation", fontsize = 20)
plt.savefig('LV1.svg', bbox_inches='tight', transparent = True, dpi=1200)
plt.show()

#%%
sd1 = np.std(bpls.bootres.y_loadings_boot[0][1])
sd2 = np.std(bpls.bootres.y_loadings_boot[1][1])
sd3 = np.std(bpls.bootres.y_loadings_boot[2][1])
sd4 = np.std(bpls.bootres.y_loadings_boot[3][1])
sd5 = np.std(bpls.bootres.y_loadings_boot[4][1])
sd6 = np.std(bpls.bootres.y_loadings_boot[5][1])
sd7 = np.std(bpls.bootres.y_loadings_boot[6][1])
sd8 = np.std(bpls.bootres.y_loadings_boot[7][1])
sd9 = np.std(bpls.bootres.y_loadings_boot[8][1])
sd10 = np.std(bpls.bootres.y_loadings_boot[9][1])
sd11 = np.std(bpls.bootres.y_loadings_boot[10][1])
sd12 = np.std(bpls.bootres.y_loadings_boot[11][1])

print(sd1)
print(sd2)
print(sd3)
print(sd4)
print(sd5)
print(sd6)
print(sd7)
print(sd8)
print(sd9)
print(sd10)
print(sd11)
print(sd12)
#%%
LVy2 = pd.read_csv(r"LVy2.csv")
LVy2.head()
df_LVy2 = pd.DataFrame(LVy2)

error = [sd1, sd2, sd3, sd4, sd5, sd6, sd7, sd8, sd9, sd10, sd11, sd12]

measure = df_LVy2['Measure'].head(12)
cov = df_LVy2['Cov'].head(12)

fig, ax = plt.subplots(figsize =(20, 12))
ax.barh(measure, cov, xerr = error, color= 'lightcoral')
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
ax.invert_yaxis()
ax.set_xlabel("Correlation", fontsize = 20)
plt.savefig('LV2.svg', bbox_inches='tight', transparent = True, dpi=1200)
plt.show()
#%%
print(bpls.varexp)


# %%
np.savetxt("Lv1_x.csv", bpls.bootres.x_weights_normed, delimiter=",")

# %%
Lv1_x = pd.read_csv(r"Lv1_x.csv")
Lv1_x.head()
df_Lv1_x = pd.DataFrame(Lv1_x)
# %%
pc = df_Lv1_x['PC'].head(12)
bsr = df_Lv1_x['BSR'].head(12)

fig, ax = plt.subplots(figsize =(10, 5))
ax.barh(pc, bsr, color= 'grey')
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
ax.invert_yaxis()
plt.axvline(2.58, color = 'black')
plt.savefig('PC_cont.svg', bbox_inches='tight', transparent = True, dpi=1200)
plt.show()
# %%
