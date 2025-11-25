#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:15:06 2025

@author: wormulon
"""


#-----------------------------------------------------------------------------#
#                               I M P O R T S 
#-----------------------------------------------------------------------------#
import numpy as np
from paw_statistics import paw_statistics
from plotter_UI import PlotterUI
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                   Plotting with UI support 
#-----------------------------------------------------------------------------#
# Note this is not relevant to our bioRxiv manuscirpt but intended to 
# facilitate easy plotting in future. This is WIP and not tested outside 
# SPYDER IDE. This allows plotting of angular analytics and PCs. Please 
# Plot properties are not fully implemented yet and might lead errors.  

paws = paw_statistics() #instantiate paw_statistics
paws.load_data_zip() #load data from a zip file 
PlotterUI(paws) #start the plotter UI



#-----------------------------------------------------------------------------#
#                   Plotting with using scripts 
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                    Contained variables
#-----------------------------------------------------------------------------#


# Instantiating the paw_statistics
paws = paw_statistics()

# this loads the data from the zip file
# navigate in the menue to data.zip and select for loading. 
paws.load_data_zip()

#alternatively you can state a static filepath
#paws.load_data_zip(filename='your/path/to/file.zip")

# all angles 
angles = paws.angles
# pts in the original configuration 
pts = paws.pts 
# normalized points 
pts_norm = paws.pts_2_pca(pts,flatten=False)

#label dataframe 
labls = paws.label_db



#-----------------------------------------------------------------------------#
#                    P C A   o n   a l l   p a w s 
#-----------------------------------------------------------------------------#


# Defining some colors for visualization 
colors = np.asarray([[0.5, 0., 0.],
[0.9529, 0.04313, .4078        ],
[0.54117647, 0.56470588, 0.88235294],

[0, .6        ,          0.2],

[0.94117647, 0.56470588, 0.54901961],

[0.95294118, 0.04313725, 0.40784314],


[0.40392157, 0.42352941, 0.3254902 ]])



paws.default_plot_props()
# normalizing, re-orienting, and flattening keypoint coordinates for PCA  
pts = paws.pts_2_pca(paws.pts,nth_point=6,generic_right=True,
                     re_zero_type='mid_line',mirror_type='mid_line')





# running PCA using paw_statistics + plotting 
categories = paws.run_pca(pts,
                          "paw_posture",colors=colors)

#PC scores can accessed this way: 
PC_scores = paws.X_reduced


#plotting group specific postures that summarize the average group posture
paws.compose_paws(range(30),names=categories)

#Optionally plots can be formatted using paw_style, and scaling: 
   
paw_style = {'xmargin':0.03, # x axis margin 
             'ymargin':0.03, # y axis margin
             'aspect':'equal', # aspect ratio, attention this is overwritten using scaling
             'axes':'on'} # axes settings

scaling = [4, # x scaling relative
           1, # y scaling relative
           6, # x scaling absolute
           12]# y scaling absolute

paws.compose_paws(range(30),names=categories,
                  paw_style=paw_style,scaling=scaling) 



#-----------------------------------------------------------------------------#
#           H Y P O T H E S I S   T E S T I N G  & P L O T T I N G
#-----------------------------------------------------------------------------#

#ensuring angles are between -pi and pi
paws.angle_range = '-pi' 
paws.all_angles()

#defining some color variables
finger_colors= [[0.33333333, 0.33333333, 1.],
         [0.54117647, 0.56470588, 0.88235294],
         [0.93333333, 1.        , 0.66666667],
         [0.94117647, 0.56470588, 0.54901961],
         [0.95294118, 0.04313725, 0.40784314]]

colors = np.array([[0.33333333, 0.33333333, 1.        ],
                   [0.95294118, 0.04313725, 0.40784314]])


# overall differences between injured and non-injured .........................

data_groups = [{'injury':0},{'injury':1,}]           #0,1    
                                     #6,7
               #{'injury':0,'treatment':'no treatment KO'},{'injury':1,'treatment':'no treatment KO'},       #8,9
               #{'injury':0,'treatment':'no treatment WT'},{'injury':1,'treatment':'no treatment WT'}]       #10,11

design_matrix = [[0,1],]

comparison_labels = ["injury-vs-control"]



for i in data_groups:
    a,p = paws.filter_data(i)
    paws.paw_plot(a,err_ang=14,offset=10,
                  headlines=finger_colors)



# pairwise hyptothesis tests FDR  
all_results,plot_data,plot_labels = paws.test_all_angles(data_groups,
                                                         design_matrix
                                                         ,CI_90=True,
                                                         angle_list=paws.angle_list_all)  # opening angle

#do the volcano plot and further filtering  
thresholded_results,corrected_results = paws.volcano_plot(all_results, 'relative_delta',
                                                          'qVal', 0.5, 0.05,figsize=[4,6])

#plot the heatmap for the pValues... 
p_V_mat = paws.reshuffle_pvs(corrected_results) 
paws.plot_pvalue_heatmap(p_V_mat,y_labels='auto',x_labels=comparison_labels,aspect_ratio=[1,6])
# plotting only the intuitive (named) angles
p_V_mat = paws.reshuffle_pvs(corrected_results,angles=paws.named_angles) 
paws.plot_pvalue_heatmap(p_V_mat, y_labels='auto',x_labels=comparison_labels,aspect_ratio=[1,7])

# This plots the paw_mapping, caxis is in revolutions--------------------------
# the index, here 1, lets you choose the from your stored paws.
paws.multi_group_paw_mapping(thresholded_results,1,caxis=[0,2.5]) 



#This will plot individual angles for all groups-------------------------------

# OPTIONAL: this determines the  ylims for each angle. Here we are going with the named_angles only.
# the ylims need to be in order of the named angels i.e. paws.named_angles
ylims = [[-50,140], #81
         [-50,100], #78
         [-20,60], #88
         [-20,60], #109
         [-0,90], #126
         [-80,40], #4
         [-50,30], #21
         [-40,20], #38
         [-50,50], #54
         [-30,70], #69
         [-60,40], #87
         [-100,60], #108
         [-100,100], #125
         [-100,70], #138
         [-50,110], #83
         [40,125], #95
         [50,150], #115
         [-180,180], #131
         [-20,110], #142
         ]

# OPTIONAL: plot_porperties (plot_props) allow tuning the plot appearance 
plot_props = {'top_ticks':'off',
'spine_right':False,
'spine_top':False,
'spine_left':True,
'spine_bottom':True,
'xlim':[0.5,2.5],
'ylim':'AUTO',
'xlabel':'',
'ylabel':'opening angle [o]',
'xtick_format': (90,7,'center','top')}    


paws.plot_n_angles(paws.named_angles,plot_data,plot_labels,colors,
                   plot_props=plot_props,figsize=(2,6),ylims=ylims)




