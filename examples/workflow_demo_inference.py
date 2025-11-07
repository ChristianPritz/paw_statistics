# -*- coding: utf-8 -*-
"""
Spyder Editor

Paw segmentation demo 
"""
#-----------------------------------------------------------------------------#
#                                I M P O R T S 
#-----------------------------------------------------------------------------#

from ImageSequenceExporter import ImageSequenceExporter
from tkinter import filedialog
import os
import numpy as np

#-----------------------------------------------------------------------------#
#                            F R O M   V I D E O S 
#-----------------------------------------------------------------------------#

#specify an output path for saving exported images and data
output_path = '/home/wormulon/Documents/accute pain images 4/'
if not os.path.isdir(output_path):
    os.mkdir(output_path)


#SETTING UP THE META DATA FOR LABELLING----------------------------------------

prefix = 'mouse_' # prefix for animals 
DOB = "24.10.25" # Day of birth default value 
# this defines your labels, each keyword is a label category, 
#each list the possible labels
metadata = {"treatment": ["CAPSAICIN_25","CAPSAICIN_2500","vehicle_control","CAPSAICIN_INJECTED","NO_INJECTION"], "animal_id": prefix, "DOB": DOB,
            "gender":["MALE","FEMALE"],"genotype":["WT","KO"],
            "strain":"C57BL/6","side": ["right", "left", "I DON'T KNOW"],
            "pain_status":["pain","no_pain","accute_pain","recovered"],
            "paw_posture":["clenched","open","closed"],
            "injection_status":["worked","unclear","failed"],
            "liquid_presence":["no","yes"],
            "orientation":["good","ok","useless","impossible"]}

# these are settings for the detector and keypoints
detector_settings = {'threshold':0.95,'connect_logic': [[0,     1],
                 [0,     3],
                 [0,     6],
                 [0,     9],
                 [0,    12],
                 [1,     2],
                 [3,     4],
                 [4,     5],
                 [6,     7],
                 [7,     8],
                 [9,    10],
                 [10,   11],
                 [12,   13],
                 [13,   14],
                 [1,     3],
                 [3,     6],
                 [6,     9],
                 [9,    12]],
                     
            'colors_ui': ['#5555ff',
                 '#8a90e1',
                 '#eeffaa',
                 '#f0908c',
                 '#f30b68',
                 '#5555ff',
                 '#8a90e1',
                 '#8a90e1',
                 '#eeffaa',
                 '#eeffaa',
                '#f0908c',
                '#f0908c',
                '#f30b68',
                '#f30b68',
                '#676c53',
                '#676c53',
                '#676c53',
                '#676c53'], 
                     
            'keypoint_names': ["heel","base_i","tip_i","base_ii",
                  "phal_ii","tip_ii","base_iii","phal_iii",
                  "tip_iii","base_iv","phal_iv","tip_iv",
                  "base_v","phal_v","tip_v"],
            "model_path":'/home/wormulon/Documents/trained models/paw_model_reduced/model_torch.pt',
            "device":'cuda'}
            



# OPTIONAL: use an UI for finding the movie 
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MOV Files", "*.MOV")])
# or coding the filepath as a string directly
video_path = '/home/wormulon/Downloads/IMG_0933.MOV'


exporter = ImageSequenceExporter(video_path, metadata,detector_settings,width=500,prefix=prefix,output_dir=output_path)


exporter.paw_stats.label_db.predicted_side.iloc[0]
exporter.paw_stats.plot_single_prediction(1)

#-----------------------------------------------------------------------------#
#                            F R O M   I M A G E S 
#-----------------------------------------------------------------------------#

# prefix = ''
# metadata = {"treatment": ["CAPSAICIN_25","CAPSAICIN_2500","vehicle_control","CAPSAICIN_INJECTED","NO_INJECTION"], "animal_id": prefix, "DOB": "000000",
#             "gender":["MALE","FEMALE"],"genotype":["WT","KO"],
#             "strain":"C57BL/6","side": ["right", "left", "I DON'T KNOW"],
#             "pain_status":["pain","no_pain","accute_pain","recovered"],
#             "paw_posture":["clenched","open","closed"],
#             "injection_status":["worked","unclear","failed"],
#             "liquid_presence":["no","yes"],
#             "orientation":["good","ok","useless","impossible"]}

# detector_settings = {'threshold':0.95,'connect_logic': [[0,     1],
#                  [0,     3],
#                  [0,     6],
#                  [0,     9],
#                  [0,    12],
#                  [1,     2],
#                  [3,     4],
#                  [4,     5],
#                  [6,     7],
#                  [7,     8],
#                  [9,    10],
#                  [10,   11],
#                  [12,   13],
#                  [13,   14],
#                  [1,     3],
#                  [3,     6],
#                  [6,     9],
#                  [9,    12]],
                     
#             'colors_ui': ['#5555ff',
#                  '#8a90e1',
#                  '#eeffaa',
#                  '#f0908c',
#                  '#f30b68',
#                  '#5555ff',
#                  '#8a90e1',
#                  '#8a90e1',
#                  '#eeffaa',
#                  '#eeffaa',
#                 '#f0908c',
#                 '#f0908c',
#                 '#f30b68',
#                 '#f30b68',
#                 '#676c53',
#                 '#676c53',
#                 '#676c53',
#                 '#676c53'], 
                     
#             'keypoint_names': ["heel","base_i","tip_i","base_ii",
#                   "phal_ii","tip_ii","base_iii","phal_iii",
#                   "tip_iii","base_iv","phal_iv","tip_iv",
#                   "base_v","phal_v","tip_v"],
#             "model_path":'/home/wormulon/Documents/trained models/paw_model_reduced/model_torch.pt',
#             "device":'cuda'}

# SET UP THE META DATA AS IN THE VIDEO EXAMPLE 

#image_path = filedialog.askdirectory(title="Select image Folder")
image_path ='/media/my_device/training data/paw open angle/unused3'
output_path = '/media/my_device/training data/paw open angle/unused2/output'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
exporter = ImageSequenceExporter(image_path, metadata,detector_settings,width=500,prefix=prefix,output_dir=output_path)

zip_path = '//home/wormulon/Documents/accute pain images 4/Exp--unused2.zip'
exporter = ImageSequenceExporter(image_path, metadata,detector_settings,width=500,prefix=prefix,output_dir=output_path,paw_stats=zip_path)



#-----------------------------------------------------------------------------#
#            P L O T T I NG   F R O M   S E G M E N T E D   P A W S
#-----------------------------------------------------------------------------#


finger_colors= [[0.33333333, 0.33333333, 1.        ],
         [0.54117647, 0.56470588, 0.88235294],
         [0.93333333, 1.        , 0.66666667],
         [0.94117647, 0.56470588, 0.54901961],
         [0.95294118, 0.04313725, 0.40784314]]


# viusalizing some of the predictions post-hoc
exporter.paw_stats.label_db.predicted_side.iloc[0]
exporter.paw_stats.plot_single_prediction(1)
paws = exporter.paw_stats

folder = 'Dump'


paws.angle_range = '-pi'
paws.all_angles()


sidecolors = [[0.95294118, 0.04313725, 0.40784314],
               [0.33333333, 0.33333333, 1.        ]]

paws.paw_plot_settings = {"offset":10,"err_ang":20,"max_n":5}
data_groups = [{'side':'left'},{'side':'right'}]
design_matrix = [[0,1]]
for i in data_groups:
    a,p = paws.filter_data(i)
    paws.paw_plot(a,err_ang=14,offset=10,folder=folder,tag='--' + str(i['side']),
                  headlines=finger_colors)




def compile_group_labels(data_groups,design_matrix):
    group_labels = []
    for i in design_matrix:
        group_labels.append(str(data_groups[i[0]]['side'])+ '-vs-' + str(data_groups[i[1]]['side']))
    return group_labels

group_labels = compile_group_labels(data_groups,design_matrix)
pm_labels = [[81,75,88,109,126,4,21,38,54,69,87,108,125,138,83,95,115,131,142],
             ['TOA','I-II','II-III','III-IV','IV-V','root_i','root_ii','root_iii','root_iv','root_v',
              'digit_ii','digit_iii','digit_iv','digit_v','base_i','base_ii','base_iii','base_iv','base_v'],group_labels,[12,8],10000]


plot_props = {'top_ticks':'off',
'spine_right':False,
'spine_top':False,
'spine_left':True,
'spine_bottom':True,'xlim':[0.5,2.5],'ylim':[50,150],'xlabel':'','ylabel':'opening angle [o]'}    


# for display
angle_list_short = [81,75,88,109,126,4,21,38,54,69,87,108,125,138,83,95,115,131,142]
angle_names = ['TOA','I-II','II-III','III-IV','IV-V','root i','root ii','root iii','root iv','root v',
 'digit ii','digit iii','digit iv','digit v','base i','base ii','base iii','base iv','base v']
angle_list = np.arange(0,153)


all_results,plot_data,plot_labels = paws.test_all_angles(data_groups, design_matrix,folder=folder,
                                  tag='-side-only',clean_CI=True,angle_list=angle_list)  # opening angle


#do the volcano plot 
thresholded_results,corrected_results = paws.volcano_plot(all_results, 'relative_delta', 'qVal', 0.5, 0.05,
                           tag='-side-only',folder=folder,figsize=[4.5,6])

name = 'volcano_plot_-side-only'
fig,ax = paws.load_plot(name,folder)
ax[0].set_xlim(-1.2,1.2)
paws.save_plot(fig,name,folder)


#plot the heatmap for the pValues... 
p_V_mat = paws.reshuffle_pvs(corrected_results,angle_list) 
paws.plot_pvalue_heatmap(p_V_mat, x_labels=None,y_labels=[],tag='-side-only-all',
                        folder=folder,aspect_ratio=[1,6])
p_V_mat = paws.reshuffle_pvs(corrected_results,angle_list_short) 
paws.plot_pvalue_heatmap(p_V_mat, x_labels=None, y_labels=angle_names,tag='--side-only-intutitive',
                        folder=folder,aspect_ratio=[1,6])

# Do the paw_mapping
paws.multi_group_paw_mapping(thresholded_results,1,'summed_angles','-side-only',folder,caxis=[0,36])


int_angles = [81,75,88,109,126,4,21,38,54,69,87,108,125,138,83,95,115,131,142]

ylims = [[40,150], #81
         [-30,60], #78
         [0,50], #88
         [0,50], #109
         [10,65], #126
         [-70,20], #4
         [-40,00], #21
         [-20,20], #38
         [-10,40], #54
         [20,70], #69
         [-30,20], #87
         [-30,20], #108
         [-25,30], #125
         [-30,50], #138
         [0,100], #83
         [60,110], #95
         [80,130], #115
         [120,160], #131
         [80,150], #142
         ]

paws.plot_n_angles(int_angles,plot_data,plot_labels,sidecolors,figsize=(2,6), 
                   tag='-side-only',folder=folder,
                   ylims=ylims,plot_props=plot_props,bins=15)









