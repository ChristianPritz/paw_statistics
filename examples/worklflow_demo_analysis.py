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
paws = paw_statistics(settings=None)

# this loads the data from the zip file
paws.load_data_zip()

#-----------------------------------------------------------------------------#
#                    P C A   o n   a l l   p a w s 
#-----------------------------------------------------------------------------#

# Defining some colors for visualization 
colors = np.asarray([[0.95294118, 0.04313725, 0.40784314],
[0.33333333, 0.33333333, 1.        ],
[0.54117647, 0.56470588, 0.88235294],

[0, .6        , 51/255],

[0.94117647, 0.56470588, 0.54901961],

[0.95294118, 0.04313725, 0.40784314],


[0.40392157, 0.42352941, 0.3254902 ]])

# Defining some colors for visualization 
colors = np.asarray([[0.5, 0., 0.],
[0.9529, 0.04313, .4078        ],
[0.54117647, 0.56470588, 0.88235294],

[0, .6        , 51/255],

[0.94117647, 0.56470588, 0.54901961],

[0.95294118, 0.04313725, 0.40784314],


[0.40392157, 0.42352941, 0.3254902 ]])



paws.default_plot_props()
# normalizing, re-orienting, and flattening keypoint coordinates for PCA  
pts = paws.pts_2_pca(paws.pts,nth_point=6,generic_left=True,
                     re_zero_type='mid_line',mirror_type='mid_line')


combos = [[0,2],[5,7],[9,10],[9,16],[10,16]]
# running PCA using paw_statistics + plotting 
categories = paws.run_pca(pts,
                          "paw_posture",colors=colors)


# plotting eigenpostures


#plotting group specific postures that summarize the average group posture




paws.compose_paws(range(30),names=categories) # probably best

#Optionally plots can be formatted using paw_style, and scaling: 
    
paw_style = {'xmargin':0.03, # x axis margin 
             'ymargin':0.03, # y axis margin
             'aspect':'equal', # aspect ratio, attention this is overwritten using scaling
             'axes':'off'} # axes settings

scaling = [4, # x scaling relative
           1, # y scaling relative
           6, # x scaling absolute
           12]# y scaling absolute

paws.compose_paws(range(30),names=categories,
                  paw_style=paw_style,scaling=scaling) 



#-----------------------------------------------------------------------------#
#           H Y P O T H E S I S   T E S T I N G  & P L O T T I N G
#-----------------------------------------------------------------------------#


folder = ''
paws.angle_range = '-pi'
paws.all_angles()

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




group_labels = []
for i in design_matrix:
    group_labels.append('injury' + str(data_groups[i[0]]['injury'])+ '-vs-' 'injury' + str(data_groups[i[1]]['injury']))



paws.paw_plot_settings = {"offset":10,"err_ang":20,"max_n":60}
for i in data_groups:
    a,p = paws.filter_data(i)
    paws.paw_plot(a,err_ang=14,offset=10,folder=folder,tag='-injury-' + str(i['injury']),
                  headlines=finger_colors)


# for display
angle_list_short = [81,75,88,109,126,4,21,38,54,69,87,108,125,138,83,95,115,131,142]
angle_names = ['TOA','I-II','II-III','III-IV','IV-V','root i','root ii','rooti ii','root iv','root v',
 'digit ii','digit iii','digit iv','digit v','base i','base ii','base iii','base iv','base v']
angle_list = np.arange(0,153)
  
all_results,plot_data,plot_labels = paws.test_all_angles(data_groups,
                                                         design_matrix
                                                         ,CI_90=True,
                                                         angle_list=angle_list)  # opening angle

#do the volcano plot 
thresholded_results,corrected_results = paws.volcano_plot(all_results, 'relative_delta', 'qVal', 0.5, 0.05,
                           tag='injury_no_subgroups_81',folder=folder,figsize=[4,6])

#plot the heatmap for the pValues... 
p_V_mat = paws.reshuffle_pvs(corrected_results,angle_list) 
paws.plot_pvalue_heatmap(p_V_mat, x_labels=None, y_labels=None,tag='-injury_no_subgroups_81-all',
                        folder=folder,aspect_ratio=[1,6])
p_V_mat = paws.reshuffle_pvs(corrected_results,angle_list_short) 
paws.plot_pvalue_heatmap(p_V_mat, x_labels=None, y_labels=angle_names,tag='-injury_no_subgroups_81-intutitive',
                        folder=folder,aspect_ratio=[1,7])

# Do the paw_mapping
paws.multi_group_paw_mapping(thresholded_results,1,'summed_angles','injury_no_subgroups_81',folder,caxis=[0,1000]) #28 or 1800

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


plot_props = {'top_ticks':'off',
'spine_right':False,
'spine_top':False,
'spine_left':True,
'spine_bottom':True,'xlim':[0.5,2.5],'ylim':[50,150],'xlabel':'','ylabel':'opening angle [o]'}    

plot_props["xlim"] = (0.5,2.5)
plot_props["xtick_format"] = (90,7,'center','top')


paws.plot_n_angles(angle_list_short,plot_data,plot_labels,colors,figsize=(2,6), 
                   tag='injury_no_subgroups_81',folder=folder,
                   ylims=ylims,plot_props=plot_props)




