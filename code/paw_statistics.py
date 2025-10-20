#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:33:04 2024

@author: Christian Pritz
"""
import os,cv2,copy,math,itertools,shutil, tempfile, zipfile,pickle
import numpy as np
import pandas as pd
import tkinter as tk
from scipy import stats
import pycircstat as circ
import scipy.spatial.distance
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from IPython import embed



class paw_statistics():
    # maybe chain up some of the parameters in a dictionary....................
    # replace all the settings == Nonene checks with key in dict checks.
    def __init__(self,settings,columns=None):
        if settings is None:
            settings = self.default_settings() 
            
        if not 'plot_path' in settings:
            self.plot_path = ''
        else:  
            self.plot_path = settings['plot_path'] 
        
        if not 'mid_point' in settings:
            self.mid_point = 6
        else:
            self.mid_point = settings['mid_point']
        
        if settings['connect_logic'] == None:
            self.connect_logic = [     [0,     1],
                 [0,     4],
                 [0,     8],
                 [0,    12],
                 [0,    16],
                 [1,     2],
                 [2,     3],
                 [4,     5],
                 [5,     6],
                 [6,     7],
                 [8,     9],
                 [9,    10],
                [10,    11],
                [12,    13],
                [13,    14],
                [14,    15],
                [16,    17],
                [17,    18],
                [18,    19],
                 [1,     4],
                 [4,     8],
                 [8,    12],
                [12,    16]]     
        else:  
            self.connect_logic = settings['connect_logic']
        
        if not 'colors' in settings:
            self.colors = np.tile([255, 0, 0],(len(self.connect_logic),1)) 
        else:  
            self.colors = settings['colors']
        
        if not 'colors_ui' in settings:
            self.colors_ui = repeated_list = ['#5555ff'] * len(self.connect_logic)
        else:  
            self.colors_ui = settings['colors_ui']
        
            
        if  not 'keypoint_names' in settings :
            self.keypoint_names = ["heel","base_i","tip_i","claw_i","base_ii","phal_ii","tip_ii","claw_ii","base_iii","phal_iii","tip_iii","claw_iii","base_iv","phal_iv","tip_iv","claw_iv","base_v","phal_v","tip_v","claw_v"]
        else:  
            self.keypoint_names = settings['keypoint_names']
            
        self.num_keypoints = np.max(self.connect_logic)+1
        #print(self.num_keypoints)
        
        if not 'ang_comps' in settings:
            self.ang_comps = list(itertools.combinations(self.connect_logic, 2))
            self.ang_comps[142] = ([12, 9], [12, 13]) #exception for the digit number 5 that has the base edge left instead of right. 
        else :
            self.ang_comps = settings['ang_comps']
            
        if not 'intresting_angles' in settings:
            self.intresting_angles = [([1,2],[4,5]),
                                      ([4,5],[8,9]),
                                      ([8,9],[12,13]),
                                      ([12,13],[16,17])]
        
        else :
            self.intresting_angles = settings['intresting_angles']
        
        self.interesting_idcs = []
        for i in self.intresting_angles:
            self.interesting_idcs.append(self.ang_comps.index(i))
        

        if not 'plt_prp' in settings:
            self.default_plot_props()
        else: 
            self.plt_prp = settings['plt_prp']
        
       
        self.session_id = settings['session_id']
        
        if not 'angle_range' in settings:
            self.angle_range = 'pi'
        else: 
            self.angle_range = settings['angle_range']
        
        


        #data containers ------------------------------------------------------
        
        #updated by add_data
        self.pts = np.empty((0,self.num_keypoints,3),dtype=float)
        self.boxes = np.empty((0,1,4),dtype=float)
        self.centers = np.empty((0,1,2),dtype=float)
        self.p_dists = np.empty((0,int((self.num_keypoints*self.num_keypoints-self.num_keypoints)/2)),dtype=float)
        
        #self.instance_identy = []
        self.stat_vector = None
        # updated by by collect_data
        if columns is None:
            self.label_db = pd.DataFrame(columns = ['genotype','gender','side','treatment','paw_posture','pain_status','image_name','paw_index','remark','animal_id','useful'])
        else: 
            self.label_db = pd.DataFrame(columns = columns)

        #updated by .all_angles()
        self.angles = np.empty((0,1,len(self.ang_comps)),dtype=float)
        
        print('[PAW STATISTICS] Initialized')
    
    def default_settings(self):
        settings = {'connect_logic': [[0,     1],
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
                    
                    'keypoint_names': ["heel","base_i","tip_i","base_ii",
                          "phal_ii","tip_ii","base_iii","phal_iii",
                          "tip_iii","base_iv","phal_iv","tip_iv",
                          "base_v","phal_v","tip_v"],
                    
                    'intresting_angles': [([1,2],[3,4]),
                            ([3,4],[6,7]),
                            ([6,7],[9,10]),
                            ([9,10],[12,13])],
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
                    'colors': [[0.33333333, 0.33333333, 1.        ],
                             [0.54117647, 0.56470588, 0.88235294],
                             [0.93333333, 1.        , 0.66666667],
                             [0.94117647, 0.56470588, 0.54901961],
                             [0.95294118, 0.04313725, 0.40784314],
                             [0.33333333, 0.33333333, 1.        ],
                             [0.54117647, 0.56470588, 0.88235294],
                             [0.54117647, 0.56470588, 0.88235294],
                             [0.93333333, 1.        , 0.66666667],
                             [0.93333333, 1.        , 0.66666667],
                             [0.94117647, 0.56470588, 0.54901961],
                             [0.94117647, 0.56470588, 0.54901961],
                             [0.95294118, 0.04313725, 0.40784314],
                             [0.95294118, 0.04313725, 0.40784314],
                             [0.40392157, 0.42352941, 0.3254902 ],
                             [0.40392157, 0.42352941, 0.3254902 ],
                             [0.40392157, 0.42352941, 0.3254902 ],
                             [0.40392157, 0.42352941, 0.3254902 ]],
                    'plot_path': '/home/wormulon/Documents/paw pain manuscript/figures/plots',
                    'plt_prp':self.default_plot_props(),
                    'session_id':'Analysis',
                    'mid_point':6,
                    'angle_range':'-pi'
                    }
        return settings
        
    
          
    
    def add_data(self,pts,bxs):
        # this only handles data one by one....................................
        if not isinstance(pts, np.ndarray):
            pts = pts.numpy()
        # convert boxes here to numpy if not yet converted. 
            
        #print('pts shape is',pts,pts.shape)   
        # processing data 
        for idx in range(0,pts.shape[0]):
            #print('bxs are:', bxs[idx],idx)
            self.pts = np.append(self.pts,pts[idx,:,:].reshape([1,self.pts.shape[1],3]),axis=0)

            if hasattr(bxs,'tensor'):
                self.boxes = np.append(self.boxes,np.reshape(bxs[idx].tensor.numpy(),(1,1,4)),axis=0)
                self.centers = np.append(self.centers,np.reshape(bxs[idx].get_centers().numpy(),(1,1,2)),axis=0)
            else:
                self.boxes = np.append(self.boxes,np.reshape(bxs[idx],(1,1,4)),axis=0)
                #print('idx and box shape and box',idx, bxs.shape, bxs)
                
                ctr = np.asarray([bxs[idx][0]+bxs[idx][2]/2,bxs[idx][1]+bxs[idx][3]/2])
                self.centers = np.append(self.centers,ctr.reshape((1,1,2)),axis=0)
    
    def add_label_col(self,content,column_name):
        if len(self.label_db) != len(content):
            raise ValueError("The length of the dataframe and the numpy array must be the same.")
        self.label_db[column_name] = content


        
#--------------------------------------------------------------------------
# math & helper functions
#--------------------------------------------------------------------------
    def lap_variance(self,image):
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       lap = cv2.Laplacian(gray, cv2.CV_64F)
       return lap.var()
    
    def save_dict_to_txt(self,data: dict, filename: str, folder: str):
        fname = self.plot_path + '/' + folder + '/' + filename + '.txt'
        with open(fname, 'w', encoding='utf-8') as f:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
                
                

    def tt_split_wrap(self,X,y,test_size=None,train_size=None, 
                      random_state=None, shuffle=True, 
                      stratify=None,indices=False):
        
        if indices:
            XOld = X
            X = np.arange(0,X.shape[0])
            
            #X = X.reshape((X.size,0))
        else :
            idx =[]
            
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = test_size,
                                                            train_size=train_size,
                                                            random_state=random_state,
                                                            shuffle=shuffle,
                         stratify=stratify)
        
        #print("Shapes are", X_train.shape,y_train.shape)
        
        if indices:
            X_train_out = XOld[X_train,:]
            X_test_out = XOld[X_test,:]
            idx = [X_train,X_test]
            
        else: 
            X_train_out = X_train
            X_test_out = X_test
            
        return X_train_out,X_test_out,y_train,y_test,idx

    def clean_CI(self,data):
        data = np.asarray(data, dtype=np.float64)
        
        # Create a copy of the data to avoid modifying the original array
        cleaned_data = np.copy(data)
        
        # Iterate over each column
        for col in range(cleaned_data.shape[1]):
            # Get the column, ignoring NaN values
            col_data = cleaned_data[:, col]
            col_non_nan = col_data[~np.isnan(col_data)]
            
            # Calculate the 5th and 95th percentiles
            lower_bound = np.percentile(col_non_nan, 5)
            upper_bound = np.percentile(col_non_nan, 95)
            
            # Mask out values outside the bounds
            mask = (col_data < lower_bound) | (col_data > upper_bound)
            cleaned_data[mask, col] = np.nan
    
        return cleaned_data
    
    def clean_CI_circ(self,data, high=0.95, low=0.05, units="radians"):
        """
        Removes outliers in each column of a 2D NumPy array of circular data
        based on the confidence interval (e.g., 5th and 95th percentiles).
    
        Args:
            data (numpy.ndarray): n x m array of angular values (in radians or degrees).
            high (float): Upper confidence interval limit (default is 0.95).
            low (float): Lower confidence interval limit (default is 0.05).
            units (str): 'radians' or 'degrees', defines the unit of input data.
    
        Returns:
            numpy.ndarray: Cleaned array with outliers replaced by NaN.
        """
        # Ensure the input is a NumPy array
       
        data = np.asarray(data, dtype=np.float64)
        
        if units not in ["radians", "degrees"]:
            raise ValueError("Units must be 'radians' or 'degrees'.")
        
        # Convert degrees to radians if necessary
        if units == "degrees":
            data = np.deg2rad(data)
        
        # Create a copy of the data to avoid modifying the original array
        cleaned_data = np.copy(data)
        
        # Iterate over each column
        for col in range(cleaned_data.shape[1]):
            # Get the column, ignoring NaN values
            col_data = cleaned_data[:, col]
            col_non_nan = col_data[~np.isnan(col_data)]
            
            # Calculate the circular mean and standard deviation
            mean = scipy.stats.circmean(col_non_nan, high=np.pi, low=-np.pi)
            std = scipy.stats.circstd(col_non_nan, high=np.pi, low=-np.pi)
            
            # Define the confidence interval boundaries
            lower_bound = mean - std * 1.96
            upper_bound = mean + std * 1.96
            
            # Normalize the angles to [-pi, pi]
            normalized_col = (col_data - mean + np.pi) % (2 * np.pi) - np.pi
            
            # Mask out values outside the bounds
            mask = (normalized_col < (lower_bound - mean)) | (normalized_col > (upper_bound - mean))
            cleaned_data[mask, col] = np.nan
        
        # Convert back to degrees if necessary
        if units == "degrees":
            cleaned_data = np.rad2deg(cleaned_data)
        
        return cleaned_data
    
    def get_average_link_distances(self):
        idx = np.multiply(self.label_db.paw_posture == 'open',self.label_db.orientation=='excellent')
        pts = self.pts[idx,:,:]
        distances = np.tile(np.nan,(np.sum(idx),len(self.connect_logic)))
        for idx,i in enumerate(pts):
            for j in range(len(self.connect_logic)):
                p1 = self.connect_logic[j][0]
                p2 = self.connect_logic[j][1]
                print(i,p2)
                x = i[p2,0] - i[p1,0]
                y = i[p2,1] - i[p1,1]
                distances[idx,j] = np.sqrt(x**2+y**2)

        norm = np.tile(distances[:,2].reshape((distances.shape[0],1)),(1,len(self.connect_logic)))
        norm_dist = np.divide(distances,norm)   
        fig,ax = plt.subplots(dpi=600)
        plt.bar(np.arange(0,18),np.divide(np.std(norm_dist,axis=0),np.mean(norm_dist,axis=0)))
        plt.show()        
        return norm_dist
    

   
    # batch functions----------------------------------------------------------
    def filter_data(self,filter_criteria,pts=None,angles=None,side_mean=False):
        #generate some logic ones... 
        #indices = self.label_db["gender"] != ''
        def filter_with_tolerance(column,value,tolerance):
            return (self.label_db[column] > value - tolerance) & (self.label_db[column] <= value + tolerance) 
            
        if isinstance(pts, type(None)):
            pts = self.pts
        if isinstance(angles, type(None)):
            angles = self.angles
        
        individuals = []
        indices = np.full((len(self.label_db),), True)
        for column, criterion in filter_criteria.items():
            if isinstance(criterion,list):
                yn = filter_with_tolerance(column,criterion[0],criterion[1])
                
            else:
                yn = self.label_db[column] == criterion
            indices = (indices) & yn
            
        
        if len(angles.shape) == 3:
            angles = angles[indices,:,:]
        else :
            angles = angles[indices,:]
        pts = pts[indices,:,:]
        
        individuals = self.label_db.animal_id[indices].to_numpy()
        
        if side_mean:
            print('AVERAGING LEFT AND RIGHT PAWS!')
            u_inds = np.unique(individuals)
            nu_angles = np.tile(np.nan,(u_inds.size,angles.shape[1],angles.shape[2]))
            nu_pts = np.tile(np.nan,(u_inds.size,pts.shape[1],pts.shape[2]))
            for idx,i in enumerate(u_inds):
                nu_pts[idx,:,:] = np.mean(pts[individuals==i,:,:],axis=0)
                radAng = np.radians(angles[individuals==i,:,:])
                degAng = np.rad2deg(scipy.stats.circmean(radAng,axis=0))
                wrapAng = ((degAng + 180) % 360) - 180
                nu_angles[idx,:,:] = wrapAng
            angles = nu_angles
            pts = nu_pts
        
        return angles,pts
        
    
    
    def all_angles(self,pts=None):
        if isinstance(pts, type(None)):
            pts = self.pts_2_pca(self.pts,nth_point=self.mid_point,generic_left=True,flatten=False)

        self.angles = np.empty((0,1,len(self.ang_comps)),dtype=float)

        for i in pts:

            if len(pts.shape) == 3:
                angs = self.matrix_pairwise_angles(i[:,0:2],self.ang_comps)
            else: 
                angs = self.matrix_pairwise_angles(i[:,2],self.ang_comps)
            self.angles = np.append(self.angles,angs.reshape([1,1,len(self.ang_comps)]),axis=0)

            
    def scale_0_1(self,array):
        return (array - np.min(array))/(np.max(array) - np.min(array))
    
    #detailed------------------------------------------------------------------
    def update_pdist(self,normalize=False):
        self.p_dists = np.empty((0,int((self.num_keypoints*self.num_keypoints-self.num_keypoints)/2)),dtype=float)
        for i in self.pts:
            dists =  scipy.spatial.distance.pdist(i[:,0:2]) 
            if normalize:
                mD = np.mean(dists[[0,3,7,11,15]])
                #mD = np.max(dists)
                dists = dists/mD
            print('dists',dists.shape)
            print('p_dists',self.p_dists.shape)
            self.p_dists = np.append(self.p_dists,dists.reshape((1,len(dists))),axis=0)
        
    
    
    
    def matrixdot(self,v1,v2):
        return np.sum(np.multiply(v1,v2),axis=1)
    def matrixlength(self,v):
        return np.sqrt(self.matrixdot(v,v))
    
    def matrixangle(self,array1, array2):
        """
        Calculate the angle between corresponding vectors in two n x 2 numpy arrays.
        
        Args:
        - array1: numpy array of shape (n, 2), each row is a 2D vector [x, y].
        - array2: numpy array of shape (n, 2), each row is a 2D vector [x, y].
        
        Returns:
        - angles: numpy array of angles (in radians) between corresponding vectors, range [0, 2*pi].
        """
        
        # Check that both arrays have the same shape
        if array1.shape != array2.shape:
            raise ValueError("Input arrays must have the same dimensions")
        
        # Calculate dot product of corresponding vectors
        dot_products = np.einsum('ij,ij->i', array1, array2)
        
        # Calculate magnitudes (norm) of vectors
        norms1 = np.linalg.norm(array1, axis=1)
        norms2 = np.linalg.norm(array2, axis=1)
        
        # Calculate cosine of the angle
        cos_theta = dot_products / (norms1 * norms2)
        
        # Clip the cos_theta values to avoid numerical errors beyond [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calculate the angle in radians between [0, pi]
        angles = np.arccos(cos_theta)
        if self.angle_range == '-pi':
            # Calculate the sign of the angle using the cross product to determine direction
            cross_products = np.cross(array1, array2)
            
            # adjusting the sign of the angle.
            angles = np.where(cross_products < 0, - angles, angles)
        
        return angles
 
    
           
    def dotproduct(self,v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))
    
    def length(self,v):
      return math.sqrt(self.dotproduct(v, v))
    
    def angle(self,v1, v2):
      return math.acos(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2)))
    def getVector(self,idcs,pts):
        return pts[idcs[1]]-pts[idcs[0]];
        
    

    def matrix_pairwise_angles(self,points,cmbnts):

        v1 = np.full((int(len(cmbnts)),2),np.nan)
        v2 = np.full((int(len(cmbnts)),2),np.nan)
        for idx,i in enumerate(cmbnts):
            v1[idx,:] = points[i[0][1]]-points[i[0][0]]
            v2[idx,:] = points[i[1][1]]-points[i[1][0]]
        angs = np.expand_dims(np.rad2deg(self.matrixangle(v1,v2)),axis=1)
        return angs  
    
    
    def transform_coordinates(self,pts,nth_point,scaling = True):
        transformed_sets = []
       
        for coords in pts:
            # rotating the coordinates. calculating vectors between one an n points
            translation_vector = coords[0]
            translated_coords = coords - translation_vector
           
            #Rotate coordinates so that the vector from point 1 to the nth is parallel with the x-axis
            vector = translated_coords[nth_point]  # Vector from point 1 to point n
            angle = np.arctan2(vector[1], vector[0])
            rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle),  np.cos(-angle)]
            ])
            rotated_coords = np.dot(translated_coords, rotation_matrix.T)
           
            #Scale coordinates so that the distance between point 1 and the nth point is 1
            if scaling:
                distance = np.linalg.norm(rotated_coords[9])
                scaled_coords = rotated_coords / distance
                transformed_sets.append(scaled_coords)
            else: 
                transformed_sets.append(rotated_coords)
                
        return np.array(transformed_sets)
        
    def pts_2_pca(self,pts,nth_point=8,generic_left=False,flatten=True,
                  scaling=True,mirror_type='mid_line',re_zero_type='mid_line'):
        #print("Converting")
        pts = pts[:,:,[0,1]]
        pca_pts = self.transform_coordinates(pts, nth_point,scaling=scaling)
        
        if generic_left:
            #print('GENERIC LEFT')
            for idx,i in enumerate(self.label_db['side']):
                if i == 'right':
                    #print('flipped')
                    if mirror_type == 'centre_of_mass':
                        pca_pts[idx,:,:] = self.mirror_on_com_axis(pca_pts[idx,:,:])
                    elif mirror_type == 'mid_line':
                        
                        pca_pts[idx,:,[1]] = pca_pts[idx,:,[1]]*-1 
                    else :
                        print('ERROR NO MIRRORING TYPE SPECIFIED--------------')
                #rezeroing on the heel 
                #print(pca_pts[idx,0,:])
                
                if re_zero_type == 'centre_of_mass':
                    CoM = np.mean(pca_pts[idx,:,:],axis=0)
                        #anker_point = np.tile(pca_pts[idx,0,:],(pca_pts.shape[1],1))
                    anker_point = np.tile(CoM,(pca_pts.shape[1],1))
                    #print('ankerpoint ',anker_point.shape,pca_pts[idx,:,:].shape)
                    pca_pts[idx,:,:] = pca_pts[idx,:,:] - anker_point
                elif re_zero_type == 'mid_line':
                    anker_point = np.tile(pca_pts[idx,0,:],(pca_pts.shape[1],1))
                    pca_pts[idx,:,:] = pca_pts[idx,:,:] - anker_point
                    
        if flatten:
            pca_pts = pca_pts.reshape((pts.shape[0],self.num_keypoints*2)) 
            
        return pca_pts
    def mirror_on_com_axis(self, pts):
        # Calculate the center of mass (CoM) in the y-direction
        y_com = np.mean(pts[:, 1])
        
        # Mirror the y-coordinates with respect to the CoM
        mirrored_pts = pts.copy()
        mirrored_pts[:, 1] = 2 * y_com - pts[:, 1]
        
        return mirrored_pts
    
    def kill_the_unworthy(self):
        #deletes any entry with label_db entry useful = "no"
        
        killDx = np.where(self.label_db.useful == "no")[0]
        self.pts = np.delete(self.pts, killDx, axis=0)
        self.angles = np.delete(self.angles, killDx, axis=0)
        self.boxes = np.delete(self.boxes, killDx, axis=0)
        self.centers = np.delete(self.centers, killDx, axis=0)
        self.label_db = self.label_db.drop(self.label_db.index[killDx]).reset_index(drop=True)
        self.consistency_check()
            
    def consistency_check(self):
        a = len(self.label_db) == self.pts.shape[0]
        b = len(self.label_db) == self.boxes.shape[0]
        c = len(self.label_db) == self.centers.shape[0]
        d = len(self.label_db) == self.angles.shape[0]
        
        chksm = a & b & c & d 
        if chksm:
            print('[INFO] Data set is consistent')
        else:
            if not a:
                print('[ERROR] Pts and labels are inconsistent!')
            if not b:
                print('[ERROR] Boxes and labels inconsistent!')
            if not c:
                print('[ERROR] Centers and  abels inconsistent!')
            if not d:
                print('[ERROR] Angles and labels inconsistent!')
        
#--------------------------------------------------------------------------
# statistics analysis functions
#--------------------------------------------------------------------------
    def adjust_pvalues(self,df, column_name,method='bh'):
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        # Perform FDR adjustment
        p_values = df[column_name]
        print(p_values)
        p_values = np.where(np.isfinite(p_values.values), p_values.values, 1.0)
        
        
        
        if method == 'bh':
            adjusted_pvalues = stats.false_discovery_control(p_values, method=method)
        else: 
            adjusted_pvalues = multipletests(p_values, method=method)[1]
        
        # Define significance stars
        def significance_stars(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns."
        
        # Add new columns to the DataFrame
        df['qVal'] = adjusted_pvalues
        df['indicator'] = df['qVal'].apply(significance_stars)
        
        return df    


    def p_values_to_stars(self,p_values):
        """
        Converts an array of p-values to star notation.

        Parameters:
        p_values (numpy array): An array of p-values.

        Returns:
        numpy array: An array of star notations corresponding to the p-values.
        """
        stars = []
        for p in p_values:
            if p < 0.001:
                stars.append('***')
            elif p < 0.01:
                stars.append('**')
            elif p < 0.05:
                stars.append('*')
            elif p < 0.1:
                stars.append('ns.')
            else:
                stars.append('ns.')
        
        return stars
    
    def test_all_angles(self,data_groups,design_matrix,watson_williams=True,
                    tag='',folder='',sorting_method=None,clean_CI=False,angle_list=None,
                        side_mean=False,permuts=10000,common_std_method='pooled'):
      
        def wrap_degree(angles):
            if self.angle_range == "-pi":
                return ((angles + 180) % 360) - 180
            elif self.angle_range == "2pi":
                return angles % 360
            elif self.angle_range == "pi":
                return angles % 180
                
        def wrap_radians(angles):
            if self.angle_range == "-pi":
                return  ((angles + np.pi) % (2 * np.pi)) - np.pi
            elif self.angle_range == "2pi":
                return angles % (np.pi*2)
            elif self.angle_range == "pi":
                return angles % np.pi
            
        def circular_weighted_mean(mean1, mean2, n1, n2, degrees=True):
            if degrees:
                mean1 = np.deg2rad(mean1)
                mean2 = np.deg2rad(mean2)
        
            # Weighted vector sum
            x = n1 * np.cos(mean1) + n2 * np.cos(mean2)
            y = n1 * np.sin(mean1) + n2 * np.sin(mean2)
        
            mean = np.arctan2(y, x)  # result in radians
        
            if degrees:
                mean = wrap_degree(np.rad2deg(mean))
  
            else:
                mean= wrap_radians(mean)
        
            return mean
        
        
        def common_std(a, b, method="combined"):
            #input is in radians 
            # output is in degree for user's sake 
            a = np.asarray(a)
            b = np.asarray(b)
        
            n1, n2 = a.size, b.size
            s1 = scipy.stats.circvar(a)   # variance in rad^2
            s2 = scipy.stats.circvar(b)
            std1, std2 = np.rad2deg(np.sqrt(s1)), np.rad2deg(np.sqrt(s2))
        
            if method == "pooled":
                sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
                return np.rad2deg(np.sqrt(sp2)), std1, std2
        
            elif method == "combined":
                mean1 = scipy.stats.circmean(a)
                mean2 = scipy.stats.circmean(b)
                mean12 = circular_weighted_mean(mean1, mean2, n1, n2, degrees=False)
        
                # Use circular distance instead of linear difference
                def circ_dist(theta1, theta2):
                    return np.angle(np.exp(1j * theta1) / np.exp(1j * theta2))
        
                s12 = ((n1 - 1) * s1 + (n2 - 1) * s2 +
                       n1 * circ_dist(mean1, mean12) ** 2 +
                       n2 * circ_dist(mean2, mean12) ** 2) / (n1 + n2 - 1)
        
                return np.rad2deg(np.sqrt(s12)), std1, std2
        
        
        
        def reduce_paws(arr,df,method):
            # Extract the column of strings
            labels = df.iloc[:, 0]
            
            # Dictionary to store sum and count for each unique item
            unique_sums = {}
            unique_counts = {}
            
            # Iterate over the DataFrame and numpy array
            for idx, label in labels.items():
                if label in unique_sums:
                    unique_sums[label] += arr[idx]
                    unique_counts[label] += 1
                else:
                    unique_sums[label] = arr[idx]
                    unique_counts[label] = 1
            
            # Create the new averaged array
            unique_sums = {}
            unique_counts = {}
            indices = {}
            
            # Iterate over the DataFrame and numpy array
            for idx, label in labels.items():
                if label in unique_sums:
                    unique_sums[label] += arr[idx]
                    unique_counts[label] += 1
                else:
                    unique_sums[label] = arr[idx]
                    unique_counts[label] = 1
                    indices[label] = idx  # Record first occurrence of the label
            
            # Create a new numpy array with averaged values
            new_array = np.zeros(len(unique_sums))
            
            # Fill the new array with averaged values
            for i, (label, total) in enumerate(unique_sums.items()):
                new_array[i] = total / unique_counts[label]
            
            return new_array
        
        
        data = np.tile(np.nan,(len(self.label_db),len(data_groups)*self.angles.shape[2]))
        result_df = pd.DataFrame(columns=['comparison_number','angle_number','group1','group2','mean1','mean2','delta','relative_delta','SD1','SD2','SD12','n1','n2','pVal','qVal','indicator'])
        angle_number = []
        counter = 0
        for j in np.arange(self.angles.shape[2]):
            
            for i in enumerate(data_groups):
                #print('criterion',i[1])
                angles,pts = self.filter_data(i[1],side_mean=side_mean)
                angs = angles[:,:,j]
                if not isinstance(sorting_method,type(None)):
                    ID = self.labels.label_db["animal_ID"]
                    angs = reduce_paws(angs,ID,sorting_method)
                
                data[0:len(angs),counter] = angs.reshape((angs.shape[0]))
                angle_number.append(j)
                counter += 1
        # There see
        rows_with_all_nans = np.isnan(data).all(axis=1)
        angle_data = data[~rows_with_all_nans]
        angle_number = np.asarray(angle_number)
        print("the shape of angle data is ",angle_data)
        if clean_CI:                                                            # move this below 
            angle_data = self.clean_CI_circ(angle_data,units="degrees")
            
        
        for j in np.arange(self.angles.shape[2]):                              # this iterates over all angles 
            idx = np.where(angle_number==j)
 
            #print('indices are',idx)
            j_dat = angle_data[:,idx]
            j_dat = j_dat.reshape((j_dat.shape[0],j_dat.shape[2]))
           
            
            for compNum,i in enumerate(design_matrix):
                
                gr1 = i[0]
                gr2 = i[1]
                
                sample1 = np.radians(j_dat[np.invert(np.isnan(j_dat[:,gr1])),gr1])
                sample2 = np.radians(j_dat[np.invert(np.isnan(j_dat[:,gr2])),gr2])
                
                
                
                grp1 = '-'.join(str(value) for value in data_groups[gr1].values())
                grp2 = '-'.join(str(value) for value in data_groups[gr2].values())
                
                
                SD12,SD1,SD2 = common_std(sample1,sample2,method=common_std_method)
                if SD12 > 2*np.max([SD1,SD2]):
                    print('OUTLIER ##########################################')
                    print("SDs are ",SD12,SD1,SD2)
                    print("means are ",scipy.stats.circmean(sample2),scipy.stats.circmean(sample1))
                
                #Watson williams is good but we lack the sample size in some cases. 
                #
                
                #angular randomization test
                print("Doing the test #########################",j,compNum)
                if watson_williams:
                    pVal,f_stats = circ.tests.watson_williams(sample1, sample2,nan_policy='omit')
                else:
                    pVal, observed_stat = self.angular_randomization_test(sample1, sample2,num_permutations=permuts)
                
                delta = np.rad2deg(scipy.stats.circmean(sample2))-np.rad2deg(scipy.stats.circmean(sample1))
                delta = wrap_degree(delta)
                
                relative_delta = delta/SD12;
                
                new_row = {'comparison_number':compNum,
                           'angle_number':j,
                           'group1':grp1,'group2':grp2,
                           'mean1':np.rad2deg(scipy.stats.circmean(sample1)),
                           'mean2':np.rad2deg(scipy.stats.circmean(sample2)),
                           'delta':delta,
                           'relative_delta':relative_delta,
                           'SD1':SD1,
                           'SD2':SD2,
                           'SD12':SD12,
                           'n1':sum(~np.isnan(angle_data[:,gr1])),
                           'n2':sum(~np.isnan(angle_data[:,gr2])),
                           'pVal':pVal}
         
                #print(new_row)
                fuse_df = pd.DataFrame(data=new_row,index=[0])
                result_df = pd.concat([result_df.astype(fuse_df.dtypes), fuse_df], ignore_index=True, sort=False)
                result_df = result_df.reset_index(drop=True)
        
        labels = []
        for i in range(len(data_groups)):
            labels.append('-'.join(str(value) for value in data_groups[i].values()))
        
        if len(result_df) > 1:
            result_df = self.adjust_pvalues(result_df, 'pVal')


        s_path = self.plot_path + '/' + folder +'/' + 'raw_satistics' + tag + '.csv'    
        
        
        result_df.to_csv(s_path, index=False)
        

        
        data_out = np.vstack((angle_number,angle_data))
        return result_df,data_out,labels
    
    def multi_group_paw_mapping(self,df,paw_num,display_type,tag,folder,caxis=None):
        comps = np.unique(df.comparison_number)
        for i in comps:
            print("The comparisons are ",i,'----------------------------')
            idx = df.comparison_number==i
            df_i = df[idx]
            tagX = tag + '-' + str(i) + '-'
            self.map_significance_on_paw(df_i,paw_num,tag=tagX,folder=folder,display=display_type,caxis=caxis)
    
    
    def plot_n_angles(self,angular_indices,angular_data,labels,colors,
                      figsize=(6,6),tag='',folder='',plot_props=None,
                      ylims=None,scatter_type='ordered',bins=10,
                      marker_size=50,spread=0.5,subset=None):
        
        angle_data = angular_data[1:angular_data.shape[0],:]
        for ct,i in enumerate(angular_indices):
            idx = np.where(angular_data[0,:]==i)
            
    
            plot_data = angle_data[:,idx]
            plot_data = plot_data.reshape((plot_data.shape[0],plot_data.shape[2]))
            if ylims is not None:
                print(idx)
                plot_props['ylim'] = ylims[ct]
            if subset is not None:
                plot_data = plot_data[:,subset]
                plabels = [labels[i] for i in subset]
    
            else:
                plabels = labels
            
            
            axObj = self.plot_grouped_values(plot_data, plabels,colors=colors,
                                             figsize=figsize,isCircular=True,
                                             plot_props=plot_props,
                                             scatter_type = scatter_type,
                                             bins=bins,marker_size=marker_size,
                                             spread=spread)
            
            print(folder)
            self.save_plot(axObj[0],"Group_scatter"+tag+'_'+str(i),folder)
    
    

    def angular_randomization_test(self,sample1, sample2, num_permutations=10000, unit='radian'):
        """
        Perform an Angular Randomization Test to compare two samples of circular data.
    
        Parameters:
        - sample1, sample2: Arrays of angular data (in radians or degrees).
        - num_permutations: Number of permutations to perform.
        - unit: 'radian' or 'degree' indicating the unit of the input data.
    
        Returns:
        - p_value: The p-value from the permutation test.
        - observed_stat: The observed difference in mean directions.
        """
        # Convert to radians if data is in degrees
        if unit == 'degree':
            sample1 = np.deg2rad(sample1)
            sample2 = np.deg2rad(sample2)
    
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
    
        # Compute the observed difference in circular means
        mean1 = scipy.stats.circmean(sample1)
        mean2 = scipy.stats.circmean(sample2)
        observed_stat = np.abs(np.angle(np.exp(1j * mean1) / np.exp(1j * mean2)))
    
        # Perform permutations
        count = 0
        for _ in range(num_permutations):
            np.random.shuffle(combined)
            perm_sample1 = combined[:n1]
            perm_sample2 = combined[n1:]
            perm_mean1 = scipy.stats.circmean(perm_sample1)
            perm_mean2 = scipy.stats.circmean(perm_sample2)
            perm_stat = np.abs(np.angle(np.exp(1j * perm_mean1) / np.exp(1j * perm_mean2)))
            if perm_stat >= observed_stat:
                count += 1
    
        p_value = count / num_permutations
        return p_value, observed_stat




    def reshuffle_pvs(self,df,angles):
        killDx = np.isin(df.angle_number,angles)
       
        angs = df.angle_number.iloc[killDx]
        comps = df.comparison_number.iloc[killDx]
        
        qs = df.qVal.iloc[killDx]
        
        
        u_angs = np.unique(angs)
        u_comps = np.unique(comps)
        
        data = np.tile(np.nan,(len(u_angs),len(u_comps)))
        
        for idx_ang,ang in enumerate(angles):
            qs_i = qs[angs==ang]
            comp_i = comps[angs==ang]
            for idx_comp,comp in enumerate(u_comps):
                data[idx_ang,idx_comp] = qs_i.iloc[np.where(comp_i==comp)[0]]
        
        return data
    
          
    
    
    def logistic_regression(self,x,y,test_size=0.2,showPlot=False,tag='',folder='',f1_metric = 'macro'):
        
        if test_size > 0 :
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        else: 
            x_train = x
            y_train = y
            
        # Create a logistic regression model
        model = LogisticRegression()

        # Fit the model
        model.fit(x_train, y_train)
        if test_size > 0 :
        # Predict using the test set
            y_pred = model.predict(x_test)
    
            # Evaluate the model (optional)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred,average=f1_metric)
            conf_matrix = confusion_matrix(y_test, y_pred)
        else:
            f1 = np.nan
            accuracy = np.nan
            conf_matrix = np.nan
 
        # Print the coefficients
        intercept = model.intercept_[0]
        coefficient = model.coef_[0][0]
        b_point = -intercept/coefficient
        if showPlot: 
            x = x.reshape((x.size,))
            y = y.reshape((y.size,))
            
            print(x)
            print(y)
            red1 = np.logical_and.reduce([y == 1,x>b_point])
            red2 = np.logical_and.reduce([y == 0,x<b_point])
            
            fig, ax = plt.subplots(dpi=600,figsize=self.plt_prp['size'])
            x_range = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
            y_prob = model.predict_proba(x_range)[:, 1]
            plt.plot(x_range, y_prob, color='blue', linewidth=1.8,
                     label='Logistic Regression Curve')
            
            
            ax.scatter(x, y, color='green', label='Data points',alpha=0.4)
            ax.scatter(x[red1], y[red1], color='red', label='Data points',alpha=0.8)
            ax.scatter(x[red2], y[red2], color='red', label='Data points',alpha=0.8)
            
            XL = ax.get_xlim()
            YL = ax.get_ylim()
            print(intercept,coefficient,b_point)
            ax.plot([b_point,b_point],YL,':',
                    color=[0, 0, 0, 0.3],zorder=1)
            
            plt.xlabel('Σed angle [°]')
            plt.ylabel('injury probability')
            ax.set_xlim(XL)
            ax.set_ylim(YL)
            #self.plt_prp['grid'] = True
            self.style_plot(plt,ax)
            #self.plt_prp['grid'] = False
            name =  'Logistic-Regression' + tag
            self.save_plot(fig, name, folder)
            # Plot the data and the logistic regression curve
       
        return coefficient,intercept,accuracy,conf_matrix,f1
    
    def single_log_reg(self,X,y,settings,folder='',tag='',show_plot=True):
       
        test_size = settings['test_size']
        cv_repeats = settings['cv_repeats']
        f1_metric = settings['f1_metric']
        
        coefficients = []
        intercepts = []
        accs = []
        conf_mats = []
        f1s = []
        
        for i in range(cv_repeats):
            if i==cv_repeats-1:
                coef,inter,acc,conf,f1 = self.logistic_regression(X,y,test_size = test_size,showPlot=show_plot,tag=tag,folder=folder,f1_metric=f1_metric)
            else: 
                coef,inter,acc,conf,f1 = self.logistic_regression(X,y,test_size = test_size,showPlot=False,tag=tag,folder=folder,f1_metric=f1_metric)
            coefficients.append(coef)
            intercepts.append(inter)
            accs.append(acc)
            conf_mats.append(conf)
            f1s.append(f1)
      
        #get a plot: 

        result = {'average coefficient':np.nanmean(coefficients),
                  'average intercepts':np.nanmean(intercepts),
                  'average accuracy':np.nanmean(accs),
                  'std acc':np.nanstd(accs),
                  'average f1':np.nanmean(f1s),
                  'std f1':np.nanstd(f1s),
                  'coefficients':coefficients,
                  'intercepts':intercepts,
                  'accuracies':accs,
                  'conf_mats':conf_mats}
        
        cmat=np.sum(np.asarray(conf_mats),axis=0)

        print(cmat)
        if show_plot:
            self.draw_confusion_matrix(cmat,tag=tag,folder=folder)
        
        return result
        
    def logistic_reg_all_angles(self,X,y,cv_repeats,int_idcs,test_size=0.2,
                                tag='',folder=''):
        
        accuracies = []
        intercepts = []
        coefficients = []
        index = np.arange(0,self.angles.shape[2],1)
        
        #y = self.label_db[column].to_numpy().reshape(-1,1)
        
        for i in index:
            x=X[:,:,i]
            i_coeffs = np.tile(np.nan,(cv_repeats,1))
            i_accs = np.tile(np.nan,(cv_repeats,1))
            i_ints = np.tile(np.nan,(cv_repeats,1))
            for j in range(0,cv_repeats) :
                i_coeffs[j],i_ints[j],i_accs[j],_,_ = self.logistic_regression(x, y.ravel(),test_size=test_size)
            
            accuracies.append(np.nanmean(i_accs))
            coefficients.append(np.nanmean(i_coeffs))
            intercepts.append(np.nanmean(i_ints))
        
        accuracies = np.asarray(accuracies)
        coefficients = np.asarray(coefficients)
        intercepts = np.asarray(intercepts)
        
        #adding the overall angles. 
        # x = np.sum(self.angles[:,:,self.interesting_idcs],axis=2)
        # for j in range(0,cv_repeats) :
        #     i_coeffs[j],i_ints[j],i_accs[j],_,_ = self.logistic_regression(x, y,test_size=test_size)
        # sel_acc = np.nanmean(i_accs)
        # sel_coeff = np.nanmean(i_coeffs)
        # sel_int = np.nanmean(i_ints)
        
        sel_acc = accuracies[int_idcs]
        sel_coeff = coefficients[int_idcs]
        sel_int = intercepts[int_idcs]
        
        
        
        
        # beta 1 graphs-------------------------------------------------------
        fig, ax = plt.subplots(dpi=600,figsize=self.plt_prp['size'])
        
        np.abs(coefficients) < np.abs(sel_coeff) 
        a = accuracies>sel_acc 
        b = np.abs(coefficients) < np.abs(sel_coeff) 
        hi = a & b 
        a  = np.ones(hi.shape, dtype=bool)
        low = a & np.invert(hi) 
        print('sums are',np.sum(hi),np.sum(low))
        
        ax.scatter(accuracies[low],np.abs(coefficients[low]),alpha=0.5)
        ax.scatter(accuracies[hi],np.abs(coefficients[hi]),color='green',alpha=0.5)
        
        XL = copy.copy(ax.get_xlim())
        YL = copy.copy(ax.get_ylim())
        
        ax.plot([sel_acc,sel_acc],YL,'--',color='red',alpha=0.5)
        ax.plot(XL,[np.abs(sel_coeff),np.abs(sel_coeff)],'--',color='red',alpha=0.5)
        ax.scatter(sel_acc,np.abs(sel_coeff))
        #this should go into a function
        
        plt.yscale('log', base=10)
        ax.set_xlim(XL)
        ax.set_ylim(YL)

        
        plt.xlabel('average accuarcy',fontsize=self.plt_prp["fontsize"])
        plt.ylabel('log average |β1|',fontsize=self.plt_prp["fontsize"])
        self.style_plot(plt,ax)
        name = 'LOG_REG-accuracy-vs-beta1'+tag
        self.save_plot(fig,name,folder)
        
        
        #beta 0 graphs.........................................................
        fig, ax = plt.subplots(dpi=600,figsize=self.plt_prp['size'])
        
        np.abs(intercepts) < np.abs(sel_int) 
        a = accuracies>sel_acc 
        b = np.abs(intercepts) < np.abs(sel_int) 
        hi = a & b 
        a  = np.ones(hi.shape, dtype=bool)
        low = a & np.invert(hi) 

        
        ax.scatter(accuracies[low],np.abs(intercepts[low]),alpha=0.5)
        ax.scatter(accuracies[hi],np.abs(intercepts[hi]),color='green',alpha=0.5)
        
        XL = copy.copy(ax.get_xlim())
        YL = copy.copy(ax.get_ylim())
        
        ax.plot([sel_acc,sel_acc],YL,'--',color='red',alpha=0.5)
        ax.plot(XL,[np.abs(sel_int),np.abs(sel_int)],'--',color='red',alpha=0.5)
        ax.scatter(sel_acc,np.abs(sel_int))
          
        #this should go into a function 
        plt.yscale('log', base=10)    
        ax.set_xlim(XL)
        ax.set_ylim(YL)
        plt.xlabel('average accuarcy',fontsize=self.plt_prp["fontsize"])
        plt.ylabel('log average |β0|',fontsize=self.plt_prp["fontsize"])
        self.style_plot(plt,ax)
        name = 'LOG_REG-accuracy-vs-beta0'+tag
        self.save_plot(fig,name,folder)
       
        best = np.argsort(accuracies)[::-1][:len(accuracies)]
      
        return accuracies,coefficients,best

    
        
    def analyze_randomForest(self,feature_importances=None,tag=''):
        if isinstance(feature_importances, type(None)):
            feature_importances = pd.DataFrame(self.rf_clf.feature_importances_,
                                            index = np.arange(0,len(self.ang_comps)),
                                            columns=['importance']).to_numpy()
            print('fetching features')
        
                        
        self.plot_paw_error_indicator(self.pts[1,:,:],
                                      feature_importances,
                                      np.tile(10,(15)),incidence='features',
                                      edgeRange=[2,8],ringRange=[200,100],
                                      tag=tag+'-summed_feature_importance')
        

        index = np.argsort(feature_importances.reshape(len(feature_importances)))[::-1]
        feature_importances = feature_importances[index]
         
        self.plot_paw_error_indicator(self.pts[1,:,:],
                                      index[0:20],
                                      np.tile(10,(15)),incidence='incidence',
                                      edgeRange=[2,8],ringRange=[200,100],
                                      tag=tag+'-incidence_of_angles')
                                                           
        return index,feature_importances
        
        
   
    def transform_for_pca(self,data):
        np.seterr(divide='ignore')
        means = np.tile(self.z_means,(data.shape[0],1))
        std = np.tile(self.z_std,(data.shape[0],1))
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.true_divide((data-means),std)
            d[d== np.inf] = 0
            d = np.nan_to_num(d)
        return self.pca.transform(d)
        
    
    def run_pca(self,input_data,label_col,n_components=3,colors=None,folder='',
                tag='',combos=None,numeric_labels=False,X_reduced=None):
        if isinstance(label_col,str):
            labels = self.label_db[label_col]
        else:
            labels = label_col
        
        
        #z-normalize the columns ------------------------------------------
        data = stats.zscore(input_data,axis=0)
        self.z_means = np.mean(input_data,axis=0)
        self.z_std = np.std(input_data,axis=0)
        
        #kill eventual nans arising from 0 variance -----------------------
        data[np.isnan(data)] = 0
        
        #data = data[:,~np.all(np.isnan(data), axis=0)]
        
        #creating label colors --------------------------------------------
        categories = np.unique(labels)
        print('categories',categories)
 
        
        #do the pca -------------------------------------------------------
            
        
        self.pca = PCA(n_components=n_components).fit(data)
        self.X_reduced = self.pca.transform(data)
        #plot the variance explained --------------------------------------
        
        #BARCHART--------------------------------------------------------------
        axObj = plt.subplots(dpi=600,figsize=(6,6))
        axObj[1].bar(np.arange(1,n_components+1),self.pca.explained_variance_ratio_)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Components')
        self.style_plot(plt, axObj[1])
        plt.show()
        name = 'PCA-variance-explained' + tag
        self.save_plot(axObj,name, folder)
        
        
        
        
        #plot the principal components ------------------------------------
        
        
        
        fig, axes = plt.subplots(n_components, 1, figsize=(6, 12),dpi=600)
        for i in range(n_components):
            
            axes[i].plot(self.pca.components_[i,:])
            axes[i].set_title(f'PC number {i+1}')
        plt.tight_layout()
         
         # Show the plot
        plt.show()
        name = 'PCA-PCs' + tag
        self.save_plot(fig,name, folder)
        
        
       
        
        self.pca_scatter(categories,labels,n_components,colors=colors,
                         X_reduced=None,numeric_labels=numeric_labels,
                         combos=combos,tag=tag,folder=folder,update_GM=True)       
        return categories
    
    def pca_scatter(self,categories,labels,n_components,colors=None,
                    X_reduced=None,numeric_labels=False, combos=None,tag='',
                    folder='',update_GM=False,plot_props=None):
        
              
        #plot the pc scatter grams ----------------------------------------
        def plot_cm(ax,x,y,labels,colors):
            unique_labels = np.unique(labels)
            centers = np.zeros((2,len(unique_labels)))
            
            for idx,i in enumerate(unique_labels):
                sub_grps = np.where(labels == i)
                xp = np.mean(x[sub_grps])
                yp = np.mean(y[sub_grps])
                ax.scatter(xp,yp,marker='+',color=[0.2,0.2,0.2],s=700,linewidth=5)
                
                ax.scatter(xp,yp,marker='+',color=colors[idx,:],s=600)
                centers[0,idx] = xp
                centers[1,idx] = yp 
            return centers
        if update_GM:
            self.pc_group_centers = np.zeros((n_components,len(categories)))
        
        if isinstance(colors,type(None)):
            colors = np.random.random((len(categories),3))
        label_colors = np.empty((0,3))
        if numeric_labels:
            label_colors = labels
            
        else:
            for idx,i in enumerate(labels):
                print(categories,i)
                itemindex = np.where(categories == i)
                #print(label_colors,'itemindex:',itemindex,'colors',colors)
                label_colors =  np.append(label_colors,colors[itemindex,:].reshape(1,3),axis=0)
                #print("IS NOT NUMERIC ------------------------------------")

        if isinstance(X_reduced,type(None)):
            X_reduced = self.X_reduced
        
        if isinstance(combos,type(None)):
            for i in range(n_components-1):
                fig,ax = plt.subplots(dpi=600,figsize=(6,6))
                if numeric_labels:
                    cax = ax.scatter(
                        X_reduced[:, i],
                        X_reduced[:, i+1],
                        c=label_colors,cmap='cool',
                        alpha = 0.7
                    )
                    fig.colorbar(cax,shrink=0.8)
                else:
                    
                    ax.scatter(
                        X_reduced[:, i],
                        X_reduced[:, i+1],
                        c=label_colors,
                        alpha = 0.7
                    )
                labelX = "PC " + str(i+1) + ' (' + str(np.round(self.pca.explained_variance_ratio_[i]*100,2))+'% of variance)' 
                ax.set_xlabel(labelX)
                #ax.xaxis.set_ticklabels([])
                labelY = "PC " + str(i+2) + ' (' + str(np.round(self.pca.explained_variance_ratio_[i+1]*100,2))+'% of variance)' 
                ax.set_ylabel(labelY)
                #ax.yaxis.set_ticklabels([])
                if not numeric_labels:
                    gm = plot_cm(ax,X_reduced[:, i],X_reduced[:, i+1],labels,colors)
                    if update_GM:
                        self.pc_group_centers[[i,i+1],:] = gm   
            
                self.style_plot(plt, ax,plot_props=plot_props)
                plt.show()
                
                
                self.save_plot(fig,'PCA-component-scatter'+ '-' + str(i+1)+'-vs-'+ str(i+1) + tag, folder )
        else :
            for i in combos:
                fig,ax = plt.subplots(dpi=600,figsize=(6,6))
                if numeric_labels:
                    
                    cax = ax.scatter(
                        X_reduced[:, i[0]],
                        X_reduced[:, i[1]],
                        c=label_colors,cmap='cool',
                        alpha = 0.7
                    )
                    fig.colorbar(cax,shrink=0.8)
                else:
                    
                    ax.scatter(
                        X_reduced[:, i[0]],
                        X_reduced[:, i[1]],
                        c=label_colors,
                        alpha = 0.7
                    )
                labelX = "PC " + str(i[0]+1) + ' (' + str(np.round(self.pca.explained_variance_ratio_[i[0]]*100,2))+'% of variance)' 
                ax.set_xlabel(labelX)
                #ax.xaxis.set_ticklabels([])
                labelY = "PC " + str(i[1]+1) + ' (' + str(np.round(self.pca.explained_variance_ratio_[i[1]]*100,2))+'% of variance)' 
                ax.set_ylabel(labelY)
                #ax.yaxis.set_ticklabels([])
                if not numeric_labels:
                    gm = plot_cm(ax,X_reduced[:, i[0]],X_reduced[:, i[1]],labels,colors)
                #    if update_GM:
                #        self.pc_group_centers[[i,i+1],:] = gm   
                
                self.style_plot(plt, ax,plot_props=plot_props)
                plt.show()
                self.save_plot(fig,'PCA-component-scatter'+ '-' + str(i[0]+1) + '-vs-'+ str(i[1]+1)+tag, folder)
        self.plot_colored_labels(categories, colors, folder=folder,tag=tag)
                
                
    
    
    
    
    
    
    def test_influential_points(self, X, y, fraction=0.1,transformY=None,normY=False):
        import statsmodels.api as sm
        """
        Identify and remove the most influential points in regression using Cook's distance.
        
        Parameters
        ----------
        X : np.ndarray
            Predictor matrix (n x p).
        y : np.ndarray
            Response vector (n,).
        fraction : float
            Fraction of data points to remove based on highest Cook's distance.
        
        Returns
        -------
        X_new : np.ndarray
            Predictor matrix with influential points removed.
        y_new : np.ndarray
            Response vector with influential points removed.
        influential_idx : np.ndarray
            Indices of removed observations.
        cooks_d : np.ndarray
            Cook's distance values for all observations.
        """
        if not isinstance(transformY,type(None)):
            transform, inverse = transformY['transformations']
            y = transform(y)
            
        if normY:
            y = scipy.stats.zscore(y)
        
        
        
        import statsmodels
        # Add intercept if not present
        if not np.allclose(X[:,0], 1):
            X_sm = sm.add_constant(X)
        else:
            X_sm = X
    
        # Fit OLS model
        model = statsmodels.api.OLS(y, X_sm).fit()
    
        # Influence measures
        influence = model.get_influence()
        cooks_d, _ = influence.cooks_distance
    
        # Rank by Cook's distance
        kill_num = int(len(y) * fraction)
        influential_idx = np.argsort(cooks_d)[::-1][:kill_num]
    
        # Remove influential points
        X_new = np.delete(X, influential_idx, axis=0)
        y_new = np.delete(y, influential_idx, axis=0)
        # fig,ax = plt.subplots(dpi=600)
        # ax.bar(range(len(cooks_d)), cooks_d)
        # plt.show()
        
        
        return X_new, y_new
    
  




    def random_undersample(self,X, y, undersample_rate=0.8, random_state=None,
                                  is_continous=False,n_bins=10):
        """
        Undersample the dataset with preference for underrepresented labels,
        ensuring that each class is represented at least once.
    
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            y (array-like): Class labels (n_samples,)
            undersample_rate (float): Fraction of the original data to keep (0 < rate <= 1)
            random_state (int or RandomState): Seed for reproducibility
    
        Returns:
            X_sampled, y_sampled: Balanced, undersampled dataset
        """
        y = np.array(y)
        if is_continous:
            print('')
           
            y = np.asarray(y)
            y_min, y_max = np.min(y), np.max(y)
            bin_width = (np.max(y) - np.min(y)) / n_bins
          # Create bin edges
            bin_edges = np.arange(y_min, y_max + bin_width, bin_width)
            y_par = np.digitize(y, bins=bin_edges, right=False) - 1
        else:  
            print('')
            y_par = y
           
        rng = check_random_state(random_state)
       
        n_total = len(y_par)
        n_target = int(np.floor(undersample_rate * n_total))
    
        if n_target < len(np.unique(y_par)):
            raise ValueError("Undersample rate too low to include all classes.")
    
        # Index by label
        
        
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(y_par):
            label_to_indices[label].append(idx)
    
        # Get class distribution
        label_counts = {label: len(indices) for label, indices in label_to_indices.items()}
        labels = np.array(list(label_counts.keys()))
        counts = np.array([label_counts[label] for label in labels])
    
        # Inverse frequency weights
        inv_freq = 1 / counts
        weights = np.array([inv_freq[np.where(labels == label)[0][0]] for label in y_par])
        weights /= weights.sum()
    
        # First ensure one sample per class
        selected_indices = []
        for label in labels:
            selected_indices.append(rng.choice(label_to_indices[label], 1)[0])
    
        remaining = n_target - len(selected_indices)
        if remaining > 0:
            remaining_indices = list(set(range(n_total)) - set(selected_indices))
            remaining_weights = weights[remaining_indices]
            remaining_weights /= remaining_weights.sum()
    
            additional_indices = rng.choice(
                remaining_indices,
                size=remaining,
                replace=False,
                p=remaining_weights
            )
            selected_indices.extend(additional_indices)
    
        selected_indices = np.array(selected_indices)
        return X[selected_indices], y[selected_indices]

        
#------------------------------------------------------------------------------
# plotting functions
#------------------------------------------------------------------------------
    def scatter_distribution(self,labels, values, bins=30, spread=0.3,tag='',
                             folder='',plot_props=None,colors = None,alpha=0.7,
                             log_y=False):
        
        if colors is None:
            colors = np.random.random((100,3))
        
        labels = np.array(labels)
        values = np.array(values)
        unique_labels = np.unique(labels)
    
        fig, ax = plt.subplots(figsize=(max(6, len(unique_labels)), 6))
    
        for i, label in enumerate(unique_labels):
            group_vals = values[labels == label]
            group_vals= np.delete(group_vals, np.isnan(group_vals))
            # Histogram to understand density
            [print(group_vals)]
            counts, bin_edges = np.histogram(group_vals, bins=bins)
            bin_indices = np.digitize(group_vals, bin_edges) - 1
            print("BIN_indices:", bin_indices,len(bin_indices))
            print("BIN_edges:", bin_edges,counts)
            x_vals = []
            y_vals = []
            bin_lengths = (counts-np.min(counts))/(np.max(counts)-np.min(counts))*spread
            for b in range(len(counts)):
                idxs_in_bin = np.where(bin_indices == b)[0]
                n = len(idxs_in_bin)
                if n > 0:
                    # Centered offsets around 0 → scaled to (-spread/2, +spread/2)
                    if n == 1:
                        x_offsets = np.asarray([0])
                    else:
                        x_offsets = np.linspace(-bin_lengths[b]/2, bin_lengths[b]/2, n)
                    #print(i,x_offsets)
                    x_vals.extend(i + x_offsets)
                    y_vals.extend(group_vals[idxs_in_bin])
    
            ax.scatter(x_vals, y_vals,c=colors[i], alpha=alpha)
            ax.plot([i-spread/2, i+spread/2],[np.mean(y_vals),np.mean(y_vals)],linewidth=5,color=colors[i]*0.4)
            ax.plot([i-spread/2.5, i+spread/2.5],[np.median(y_vals),np.median(y_vals)],linewidth=3,color=colors[i]*0.4)
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(unique_labels,rotation=90)
        if log_y:
            ax.set_yscale('log')
        plt.tight_layout()
        
        
        self.style_plot(plt,ax,plot_props=plot_props)
        name = 'scatter_distribution' + tag
        self.save_plot(fig,name,folder)
 
    def plot_colored_labels(self,labels, colors,folder='',tag=''):
        """
        Plots each string in `labels` with its corresponding RGB color in `colors`.
        
        Args:
            labels (list or np.array): Array of strings (length n)
            colors (np.array): n x 3 array of RGB values (floats in [0,1])
        """
        n = len(labels)
        fig,ax, = plt.subplots(figsize=(6, 0.4 * n), facecolor='white')
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.text(0.1, 1 - i / n, label, fontsize=12, color=color, va='top')
    
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        name = tag + '_legend'
        self.save_plot(fig,name,folder)
    
    def plot_xy(self,x,y,colors=None,alpha=0.5,error=None,tag='',folder='',plot_props=None,show_plot=True):
        
        if isinstance(colors,type(None)):
            colors = np.random.random((100,3))
            colors[0,:] = np.asarray([0,0,1])
        if isinstance(plot_props,type(None)):
            'defining the plot_props'
            plot_props = {'xlabel':'','ylabel':'','figsize':(6,6),'dpi':600,
                          'xticks':[], 'yticks':[],'ylim':[],'xlim':[],'logX':''}
        fig,ax = plt.subplots(figsize=plot_props["figsize"],dpi=plot_props["dpi"])
        
        if len(y.shape) > 1:
            for i in range(0,y.shape[0]):
                
                
                ax.plot(x,y[i,:],color=colors[i,:])
                if not isinstance(error,type(None)):
                    print('error')
                    upper = y[i,:] + error[i,:]
                    lower = y[i,:] - error[i,:]
                    ax.fill_between(x, lower, upper, color=colors[i,:], alpha=alpha)
            
        else:
            if len(colors.shape)>1:
                color= colors[0,:]
            else:
                color = colors
            ax.plot(x,y,color=color)
        
     
         
            if not isinstance(error,type(None)):
                print('error')
                upper = y + error
                lower = y - error
                ax.fill_between(x, lower, upper, color=colors, alpha=alpha)
       
        if len(plot_props["xticks"]) > 0:

            ax.set_xticks(plot_props["xticks"])
        if len(plot_props["yticks"]) > 0:

            ax.set_yticks(plot_props["yticks"])
        if len(plot_props["ylim"]) > 0:

            ax.set_ylim(plot_props["ylim"]) 
        if len(plot_props["xlim"]) > 0:
         
            ax.set_xlim(plot_props["xlim"])  
        if not plot_props["logX"]=='':
            ax.set_xscale(plot_props["logX"])
        
        plt.grid(True)

        plt.tight_layout()
        self.style_plot(plt,ax,show=show_plot)
        if show_plot:
            plt.show()
        
        self.save_plot(fig,tag,folder)
        
        

    def label_visualizer(self,df,columns):
        def plot_column_distribution(df, column_name):
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found in DataFrame.")
                return
        
            # Count the values
            value_counts = df[column_name].value_counts().sort_values(ascending=False)
        
            # Set figure size and style
            plt.figure(figsize=(10, 6))
            bars = plt.bar(value_counts.index.astype(str), value_counts.values, color="#4C72B0", edgecolor="black")
        
            # Add title and labels
            plt.title(f"Distribution of '{column_name}'", fontsize=16)
            plt.xlabel(column_name, fontsize=14)
            plt.ylabel("Count", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
            # Annotate bars with counts
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + max(value_counts.values)*0.01,
                         str(height), ha='center', va='bottom', fontsize=10)
        
            plt.tight_layout()
            plt.show()
        for i in columns:
            plot_column_distribution(df, i)
        

        
    def plot_variability_over_time(self,X,labels,timepoints,tps,colors,tag='',
                                   folder=''):
        #tps = np.unique(timepoints)
        lbs = np.unique(labels)
        data = np.tile(np.nan,(lbs.size,len(tps)))
        
        for t,i in enumerate(tps):
            
            idx = timepoints == i
            vrs = []
            mns = []
            for j in lbs:
                gdx = labels == j
                dx = np.multiply(idx,gdx)
                vrs.append(np.var(X[dx]))
                mns.append(np.mean(X[dx]))
                
            vrs = np.asarray(vrs)
            o_var = np.var(mns)
            vrs = vrs/o_var
            for g,j in enumerate(vrs):
                data[g,t] = j
        #plotting 
        fig,ax = plt.subplots(dpi=600,figsize=(6,3))
        for i in range(0,data.shape[0]):
            ax.plot(tps,data[i,:],color=colors[i,:],marker='o')
        
        
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Variance ratio")
        ax.set_xscale('log')
        self.style_plot(plt,ax)
        name = 'Variability_over_time-' + tag
        self.save_plot(fig,name,folder) 
        return lbs
    
    
    def instantiate_plot_props(self,plot_props=None):
        # default_props = {
        #     'ylim': [], # do not use 'AUTO'
        #     'xlim': [],
        #     'xlabel': '',
        #     'ylabel': '',
        #     'linewidth': 1.5,
        #     'fontweight': 'normal',
        #     'fontweight_ax': 'bold',
        #     'fontsize': 14,
        #     'fontsize_ax': 14,
        #     'fontname': 'DejaVu Sans Mono',
        #     'legend': 'off',
        #     'top_ticks': 'on',
        #     'spine_right': True,
        #     'spine_top': True,
        #     'spine_left': True,
        #     'spine_bottom': True,
        #     'size': (6, 6),
        #     'figsize':(6, 6),
        #     'dpi': 600,
        #     'grid': False,
        #     'tick_length': 6,
        #     'xticks':[],
        #     'yticks':[],
        # }
        default_props= {'linewidth':2.5,
                        'fontweight':'normal',
                        'fontweight_ax':'bold',
                        'fontsize':22,
                        'fontsize_ax':24,
                        'fontname':'Abyssinica SIL',#'DejaVu Sans Mono',
                        'legend':'off',
                        'top_ticks':'on',
                        'spine_right':True,
                        'spine_top':True,
                        'spine_left':True,
                        'spine_bottom':True,
                        'size':(6,6),
                        'dpi':600,
                        'grid':False,
                        'tick_length':6,
                        'xtick_format':[0,22,'center','top'],
                        'ytick_format':[0,22,'right','center']}
    
        if plot_props is None:
            plot_props = default_props.copy()
        else:
            for key, value in default_props.items():
                if key not in plot_props:
                    plot_props[key] = value
    
        return plot_props    
        
    def plot_pvalue_heatmap(self,p_values, x_labels=None, y_labels=None,tag='',
                            folder='',aspect_ratio=[10,6]):
        """
        Plots a heatmap of p-values with log10 scaling using Matplotlib.
    
        Parameters:
        - p_values (numpy.ndarray): A 2D array of p-values.
        - x_labels (list, optional): Labels for the columns (X-axis).
        - y_labels (list, optional): Labels for the rows (Y-axis).
        """
        xlim = [-0.5, p_values.shape[1]-0.5]
        ylim = [-0.5, p_values.shape[0]-0.5]
        plot_props = {'top_ticks':'on',
        'spine_right':True,
        'spine_top':True,
        'xlim':xlim,
        'ylim':ylim}
        pVs = copy.copy(p_values)
        if not isinstance(p_values, np.ndarray) or p_values.ndim != 2:
            raise ValueError("p_values must be a 2D NumPy array.")
        #flipping ylabels because in heatmap (image) y-axis is flipped
        #y_labels = np.flip(y_labels)
    
        # trimming the pV-Matrix for display to exlcude -inf after log.  
        
        pVs[pVs<1e-5] = 1e-5
        not_sig = pVs>0.05
        # Convert p-values to log10 scale
        log_p_values = -np.log10(pVs)
        log_p_values[not_sig] = -np.inf
        # Set the color scale range (log10 of 0.05 to 10E-5)
        vmin, vmax = -np.log10(0.05), -np.log10(1e-5)
    
        # Create figure and axis
        fig, ax = plt.subplots(figsize=aspect_ratio,dpi=600)
    
        # Plot heatmap
        cax = ax.imshow(log_p_values, cmap="Reds", aspect="auto", vmin=vmin, vmax=vmax)
    
        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.25, pad=0.1)
        
        tick_values = [-np.log10(0.05), -np.log10(0.01), -np.log10(0.001), -np.log10(1e-4), -np.log10(1e-5)]
        #tick_labels = ["0.05", "e1-2", "e1-3", "e1-4", "e1-5"]
        tick_labels = ["0.05", r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$']

        # Set custom ticks and labels
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels,fontsize=self.plt_prp["fontsize"]*0.9)
 
       
    
        # Set X and Y labels
        ax.set_xticks(np.arange(p_values.shape[1]))
        ax.set_yticks(np.arange(p_values.shape[0]))
    
        if not isinstance(x_labels,type(None)):
            print(x_labels)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        if not isinstance(y_labels,type(None)):
            ax.set_yticklabels(y_labels, fontsize=10)
        else:
            ax.set_yticklabels([], fontsize=10)
            ax.set_yticks([])
    
        # Label axes
        ax.set_xlabel("Comparisons" if x_labels is None else "")
        ax.set_ylabel("Angles" if y_labels is None else "")
        
    
        self.style_plot(plt, ax,plot_props=plot_props)
        name = tag + 'significance_map' 
        self.save_plot(fig,name,folder)

        
    
        # Show plot
        plt.tight_layout()
        plt.show()
        return log_p_values
    



    def plot_data_over_time(self,grp_id=None,y_data=None,colors=None,time_stamps=None,
                            tag ='',folder='',CI=False,is_angular=False,
                            xlim=None,ylim=None,plot_props=None, 
                            bin_values = None):
     
        def match_closest_bin(bin_width,values):
            #print(values)
            a = np.arange(0,np.max(values)+bin_width,bin_width)
            diff = np.abs(values[:, None] - a[None, :])
            matched_indices = np.argmin(diff, axis=1) 
            matched_values = a[matched_indices]
            return matched_values
        
        def plot_time(x,y,color,ax,plot_props):
            
            
            
            if CI:
                if is_angular:
                    y = self.clean_CI_circ(y)
                else:
                    y = self.clean_CI(y)
             
        
            
            
            if is_angular:
                means = np.rad2deg(np.nanmean(np.deg2rad(y),axis=0))
            else:
                means = np.nanmean(y,axis=0)
            
            
            
            
            if plot_props["line_mean"]:
                ax.plot(x, means, marker='+', color=color, linestyle='-', markersize=10)
            #print('means are ',y)
            
            if is_angular:
                y_mean = np.rad2deg(stats.circmean(np.deg2rad(y), axis=0,nan_policy='omit'))
                y_std = np.rad2deg(stats.circstd(np.deg2rad(y), axis=0,nan_policy='omit'))
            else:
                y_mean = np.nanmean(y, axis=0)
                y_std = np.nanstd(y, axis=0)
            
            num = np.sum(np.invert(np.isnan(y)),axis=0)
            y_sem = np.divide(y_std,num)
    
            #print(y_sem,'------------------------------')
            # Choose the metric
            metric = plot_props["metric"]
            if metric == 'STD':
                y_error = y_std
            elif metric == 'SEM':
                y_error = y_sem
            else:
                raise ValueError("Metric must be 'STD' or 'SEM'")
    
            # Plot the mean line
            ax.plot(x, y_mean, color=color, label='Mean')
    
            # Plot the shaded region
            ax.fill_between(x.flatten(), 
                             y_mean - y_error, 
                             y_mean + y_error, 
                             color=color, 
                             alpha=0.3, label=f'±1 {metric}')
            
            if plot_props["scatter"]:
                for i in range(0,x.size):
                    #print(i)
                    col = y[:,i] 
                    t = np.tile(x[i],col.shape)
                      
                    #t = np.tile(x[i],(col.size,1)) ++ (np.random.random((col.size,1))*100-50) + offset
                    #print(t.shape,col.shape)
                    ax.scatter(
                        t, col,
                        edgecolor=color,       
                        facecolor='none',       
                        marker=plot_props["scatter_symbol"],
                        alpha=0.25,
                        s=100                   
                        )
            
        if isinstance(plot_props,type(None)):
            plot_props = {'metric':'STD','scatter':True,'scatter_symbol':'^','line_mean':True,
                          'xlabel':'','ylabel':'','figsize':(6,6)}
            
        # filter the data 
        if isinstance(colors,type(None)): #if nothing is stated we plot the opening angle by default. 
            colors = np.random.random((100,3))
        
        
        if isinstance(y_data,type(None)): #if nothing is stated we plot the opening angle by default. 
            y_data = self.angles
            y_data = y_data[:,:,81] 
        
        if isinstance(grp_id,type(None)):
            grp_id = self.label_db.treatment #if nothing is stated we go by treatment. 
        if isinstance(time_stamps,type(None)):
            time_stamps = np.asarray(round(self.label_db.timestamp/1000)).reshape((len(self.label_db.timestamp),1))
        
        uni_groups = np.unique(grp_id)

        # equalizing the time stamps to seconds. #   
        
        
        timepoints = np.unique(time_stamps)
        if not isinstance(bin_values,type(None)):
            timepoints = match_closest_bin(bin_values, timepoints)
        
        print('################## consistency')
        
        fig,ax = plt.subplots(dpi=600,figsize=plot_props["figsize"])
        #print(uni_groups)
        # assembling the data into n x timepoints matrix for each group. 
        for grpdx,i in enumerate(uni_groups):
            #print('the i is',i,i[0])
            
            idcs = grp_id==i 
            #idcs = np.where(grp_id==i)
            
            
            #print(len(idcs[0]))
            #print(y_data.shape)
            #raw_y = y_data[idcs,:]
            #raw_time = time_stamps.reshape((time_stamps.size,1))[idcs,:]
            
            raw_y = y_data[idcs]
            #raw_time = time_stamps.reshape((time_stamps.size,1))[idcs]
            raw_time = time_stamps[idcs]
            #print('################', grp_id, len(idcs), grpdx)
            
            
            
            
            if not isinstance(bin_values,type(None)):
                raw_time = match_closest_bin(bin_values, raw_time)
            
            ys = np.tile(np.nan,(raw_y.size,timepoints.size))
            t = copy.copy(timepoints)
            
            for idx,j in enumerate(timepoints):
                # now sort the raw_ys into ys. and plot inside the loop
                
            
                
                
                loc = np.where(raw_time == j)
                dat = raw_y[loc]
                
                print('this is ',j,loc,dat.size)
                if dat.size > 0: 
                    ys[0:dat.size,idx] = dat.reshape((dat.size,))
                
            #print('the dat is',dat,ys.shape)
            
            killDx = np.where(np.sum(np.isnan(ys),axis=0)>=ys.shape[0]-1)
            #print(ys.shape)
            if len(killDx[0]) > 0:
                #print('killing it')
                #print(killDx[0])
                ys = np.delete(ys,killDx[0],axis=1)
                t = np.delete(t,killDx[0])
            
            
            
                
            plot_time(t,ys,colors[grpdx,:],ax,plot_props)
            
            
        if not isinstance(xlim,type(None)):
            ax.set_xlim(xlim)
        
        if not isinstance(ylim,type(None)):
            ax.set_ylim(ylim)
        name = 'data_over_time_'+tag
        ax.set_xlabel(plot_props["xlabel"])
        ax.set_ylabel(plot_props["ylabel"])
        self.style_plot(plt, ax,plot_props=plot_props)
        
        self.save_plot(fig,name,folder)
        return uni_groups



    def plot_grouped_values(self,data, groups,colors = None,spread=0.5,
                            showMedian=True,showMean=True,isCircular=False,
                            plot_props=None,figsize=None,bins=10,
                            scatter_type = 'ordered',marker_size=50):

        if plot_props is None:
            plot_props = self.instantiate_plot_props(plot_props)
        
        num_groups = len(groups)
        if isinstance(figsize, type(None)):
            figsize = (10,6)
            print('No size defined')
        
        # Create figure
        #print('figsize',figsize)
        fig,ax = plt.subplots(dpi=600,figsize=figsize)
        
        # Set some visual parameters
          # controls how much we jitter the points
        if isinstance(colors, type(None)):
            
            colors = np.random.random((num_groups,3))# Get distinct colors for each group
        
        if isinstance(colors,list):
            colors = np.asarray(colors)
        # Loop over each group
        for i, group in enumerate(groups):
            # Extract values for this group
            group_vals = data[:, i]
            
            group_vals = group_vals[np.invert(np.isnan(group_vals))]
             
            if scatter_type == 'ordered':
                counts, bin_edges = np.histogram(group_vals, bins=bins)
                bin_indices = np.digitize(group_vals, bin_edges) - 1
                x_vals = []
                y_vals = []
                bin_lengths = (counts-np.min(counts))/(np.max(counts)-np.min(counts))*spread
                print(counts.size,bin_lengths.size,counts.size==bin_lengths.size)
                
                unique_indices = np.unique(bin_indices)

                for b in unique_indices:
                     idxs_in_bin = np.where(bin_indices == b)[0]
                     n = len(idxs_in_bin)
                     if n > 0:
                         # Centered offsets around 0 → scaled to (-spread/2, +spread/2)
                         if n == 1:
                             x_offsets = np.asarray([0])
                         else:
                             
                             x_offsets = np.linspace(-bin_lengths[b]/2, bin_lengths[b]/2, n)
            
                         
                         x_vals.extend(i + x_offsets+1)
                         y_vals.extend(group_vals[idxs_in_bin]) 

            
            elif scatter_type == 'random':

                kde = scipy.stats.gaussian_kde(group_vals)
                density_values = kde(group_vals)  # Get the density estimate for each value
                #print('density is',density_values)
                density_values = (density_values-np.min(density_values))/(
                                    np.max(density_values)-np.min(density_values))
                                    
                
                # Apply jitter proportional to the inverse of the density
                x_vals = (np.random.rand(len(group_vals)) - 0.5) * spread * density_values
                y_vals = group_vals
                x_vals = np.full_like(y_vals, i + 1) + x_vals
                
            # Plot mean as a horizontal line
            if isCircular:

                mean = np.rad2deg(scipy.stats.circmean(np.radians(y_vals)))
                med = np.rad2deg(circ.median(np.radians(y_vals)))  
                
                mean = ((mean + 180) % 360) - 180
                med =  ((med + 180) % 360) - 180
                
            else:

                mean = np.mean(y_vals)
                med = np.median(y_vals)
                
            if showMean:
                
                
                ax.plot([i + 1 - 0.4, i + 1 + 0.4], 
                         [mean,mean], 
                         linewidth=4,zorder=1,color=[0, 0, 0, 0.6],
                         solid_capstyle='round')
            if showMedian:
                ax.plot([i + 1 - 0.3, i + 1 + 0.3], 
                         [med, med], 
                         linewidth=3,zorder=1,color=[0, 0, 0, 0.3],
                         solid_capstyle='round')
   
            # Scatter plot with jitter on x-axis
            x_vals = np.array(x_vals)
                      
            ax.scatter(x_vals, y_vals, 
                        color=colors[i,:], s=marker_size, alpha=0.6,zorder=3)
            
            
        
        # Customize the plot
        if not plot_props['ylim'] == 'AUTO' and len(plot_props['ylim']) > 0:
            ax.set_ylim(plot_props['ylim'])
            #print('set ylim',plot_props['ylim'])
        if not plot_props['xlim'] == 'AUTO' and len(plot_props['xlim']) > 0:
            ax.set_xlim(plot_props['xlim'])
            #print('set xlim',plot_props['xlim'])
        else:
            ax.set_xlim(0.5, num_groups + 0.5)
            
  
            
        ax.set_xlabel(plot_props['xlabel'])
        ax.set_ylabel(plot_props['ylabel'])
        ax.set_xticks(np.arange(1, num_groups + 1), groups, rotation=55)
        plt.gca().tick_params(width=2, labelsize=16, direction='out')

        self.style_plot(plt, ax,plot_props=plot_props)    
        plt.show()
        return (fig,ax)



    def draw_confusion_matrix(self,conf_mat,relative=True,tag='',folder='',fontsize=18):
        
        if relative:
            conf_mat = conf_mat/np.tile(np.sum(conf_mat,axis=1),
                                        (conf_mat.shape[0],1)).transpose()
        
        fig, ax = plt.subplots(dpi=600,figsize=(6, 6))
        plt.ylabel('ground truth')
        plt.xlabel('prediction')
        cax = ax.matshow(conf_mat, cmap='OrRd')
        #self.style_plot(plt, ax)
        ax.set_xlabel('predicted',fontsize=self.plt_prp['fontsize'],
                      fontweight=self.plt_prp['fontweight'] )       
        ax.set_ylabel('actual',fontsize=self.plt_prp['fontsize'],
                      fontweight=self.plt_prp['fontweight'] )  
        
        #fig.colorbar(cax,shrink=0.8) #,shrink=0.6)
        
        cbar = fig.colorbar(cax, shrink=0.8)         # get the Colorbar object
        cbar.ax.tick_params(labelsize=fontsize*0.8)  
        
        
        for i in range(conf_mat.size):
            c,r = np.unravel_index(i, conf_mat.shape)
            number = conf_mat[c,r]
            if number < (np.max(conf_mat)+np.min(conf_mat))/2:
                color = 'red'
            else: 
                color = 'white'
            ax.text(r, c, str(round(number,2)),horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=fontsize,fontweight='bold',color=color)
        #fig.colorbar(cax)
        #ax.axis("off")
        cax.set_clim(0, 1)
        old_prop = self.plt_prp["xtick_format"][3]
        self.plt_prp["xtick_format"][3] = 'bottom'
        self.style_plot(plt, ax)
        self.plt_prp["xtick_format"][3] = old_prop
        plt.show()
        if tag != '':
            self.save_plot(fig,'Confusion_matrix'+tag,folder)
            
        
    def compare_preds_true(self,predictions,y,tag='',folder='',plot_props=None):
        
        if isinstance(plot_props,type(None)):
            'defining the plot_props'
            plot_props = {'xlabel':'','ylabel':'','figsize':(6,6),'dpi':600,
                          'xticks':[], 'yticks':[],'ylim':[],'xlim':[]}

        fig,ax = plt.subplots(dpi=600,figsize=(6,6))
        ax.scatter(y,predictions)
        ax.get_xlim()
        XL = ax.get_xlim()
        YL = ax.get_ylim()
        lo = [XL[0],YL[0]]
        hi = [XL[1],YL[0]]
  
        if len(plot_props["xlim"]) > 0:
            ran = [plot_props["xlim"][0],plot_props["xlim"][1]]
        else:
            ran = [np.min(lo),np.max(hi)]
        ax.set_xlim(ran)
        ax.set_ylim(ran)

        x_o = np.arange(ran[0],ran[1],(ran[1]-ran[0])/1000)
        ax.plot(x_o,x_o,color=[0,0,0,1])
        ax.set_xlabel('ground truth values')
        ax.set_ylabel('predicted values')
        if len(plot_props["xticks"]) > 0:

            ax.set_xticks(plot_props["xticks"])
        if len(plot_props["yticks"]) > 0:

            ax.set_yticks(plot_props["yticks"])
        if len(plot_props["ylim"]) > 0:

            ax.set_ylim(plot_props["ylim"]) 
        if len(plot_props["xlim"]) > 0:
         
            ax.set_xlim(plot_props["xlim"]) 
        
        #self.plt_prp['grid'] = True
        self.style_plot(plt,ax)
        #self.plt_prp['grid'] = False
        ax.set_aspect(1)
        plt.show()
        if tag != '':
            self.save_plot(fig,'Preds-vs-actual'+tag,folder)
        
    def compare_preds_true_heatmap(
        
        self, predictions, y, tag='', folder='', plot_props=None, cv_data=None,
        bw_adjust=0.5, cmap='Reds', alpha=0.5
        ):
        """
        Plot predicted vs ground truth values with a smooth heatmap
        (2D Gaussian KDE) of cross-validation predictions behind the scatter.
        """
        import matplotlib
        # ---- Default plotting properties ----
        if plot_props is None:
            plot_props = {
                'xlabel': 'ground truth values',
                'ylabel': 'predicted values',
                'figsize': (6, 6),
                'dpi': 600,
                'xticks': [],
                'yticks': [],
                'ylim': [],
                'xlim': []
            }
    
        fig, ax = plt.subplots(dpi=plot_props["dpi"], figsize=plot_props["figsize"])
    
        # ---- Determine overall range (based on all data) ----
        all_vals = np.concatenate([predictions, y])
        overall_min, overall_max = np.min(all_vals), np.max(all_vals)
        margin = 0.05 * (overall_max - overall_min)
        xmin, xmax = overall_min - margin, overall_max + margin
        ymin, ymax = xmin, xmax  # symmetric for aesthetics
    
        # ---- Process cross-validation data ----
        if cv_data is not None and len(cv_data) > 0:
            cv_preds, cv_gts = [], []
            for pred, gt in cv_data:
                cv_preds.extend(pred)
                cv_gts.extend(gt)
            cv_preds, cv_gts = np.array(cv_preds), np.array(cv_gts)
    
            # Clean up NaNs
            mask = ~np.isnan(cv_preds) & ~np.isnan(cv_gts)
            cv_preds, cv_gts = cv_preds[mask], cv_gts[mask]
    
            # ---- Compute smooth 2D KDE ----
            values = np.vstack([cv_gts, cv_preds])
            kde = scipy.stats.gaussian_kde(values, bw_method=bw_adjust)
    
            # Evaluate KDE on grid that covers full plot range
            xi, yi = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
            zi = np.reshape(kde(np.vstack([xi.ravel(), yi.ravel()])), xi.shape)
    
            # ---- Heatmap ----
            im = ax.imshow(
                np.rot90(zi),
                cmap=cmap,
                extent=[xmin, xmax, ymin, ymax],
                alpha=alpha,
                aspect='auto',
                norm=matplotlib.colors.Normalize(vmin=zi.min(), vmax=zi.max())
            )
    
            # ---- Compact colorbar with only min/max ----
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_ticks([zi.min(), zi.max()])
            #cbar.ax.set_yticklabels([f"{zi.min():.2e}", f"{zi.max():.2e}"])
            cbar.ax.set_yticklabels([])
            #cbar.set_label("CV density", rotation=270, labelpad=15)
    
        # ---- Scatter of actual predictions ----
        ax.scatter(y, predictions, color='crimson', s=10, alpha=0.9, label='Full model predictions')
    
        # ---- Axes scaling ----
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    
        # ---- Diagonal (ideal prediction) ----
        x_o = np.linspace(xmin, xmax, 1000)
        ax.plot(x_o, x_o, color=[0, 0, 0, 0.7], linewidth=1, linestyle='--', label='Ideal')
    
        # ---- Labels and ticks ----
        ax.set_xlabel(plot_props["xlabel"] or "ground truth values")
        ax.set_ylabel(plot_props["ylabel"] or "predicted values")
    
        if len(plot_props["xticks"]) > 0:
            ax.set_xticks(plot_props["xticks"])
        if len(plot_props["yticks"]) > 0:
            ax.set_yticks(plot_props["yticks"])
    
        # ---- Styling ----
        ax.set_aspect(1)
        self.style_plot(plt, ax)
        plt.tight_layout()
        plt.show()
    
        # ---- Optional saving ----
        if tag != '':
            self.save_plot(fig, 'Preds-vs-Actual' + tag, folder)
            
    
    
    
    def compare_preds_true2(self, predictions, y, tag='', folder='', plot_props=None, cv_data=None,widths=7):
        
        # ---- Default plotting properties ----
        if isinstance(plot_props, type(None)):
            plot_props = {
                'xlabel': 'ground truth values',
                'ylabel': 'predicted values',
                'figsize': (6, 6),
                'dpi': 600,
                'xticks': [],
                'yticks': [],
                'ylim': [],
                'xlim': []
            }
    
        fig, ax = plt.subplots(dpi=plot_props["dpi"], figsize=plot_props["figsize"])
        print('####################### [CV DATA] #######################')
        print(cv_data)
        # ---- Process cross-validation data ----
        if cv_data is not None and len(cv_data) > 0:
            cv_preds, cv_gts = [], []
            for pred, gt in cv_data:
                cv_preds.extend(pred)
                cv_gts.extend(gt)
            cv_preds, cv_gts = np.array(cv_preds), np.array(cv_gts)
    
            print('###################[SHAPES]###########################')
            
            unique_gts = np.unique(cv_gts)
            print(unique_gts)
            for val in unique_gts:
                
                preds_at_val = cv_preds[cv_gts == val]
                print(val,preds_at_val.size)
                if preds_at_val.size > 0:
                    print("plotting",val,np.mean(preds_at_val))
                    preds_at_val[np.isnan(preds_at_val)] = 0
                    ax.violinplot(preds_at_val[np.invert(np.isnan(preds_at_val))],
                                  positions=[val], widths=widths,
                                  showmeans=False, showextrema=False,
                                  showmedians=False)
            for pc in ax.collections[-len(unique_gts):]:
                pc.set_facecolor((0.9, 0.2, 0.5, 0.5))  # light blue semi-transparent
                pc.set_edgecolor('none')
    
        # ---- Scatter of actual predictions ----
        ax.scatter(y, predictions, color='darkred', s=10, alpha=0.8, label='Predictions')
    
        # ---- Axes scaling ----
        XL = ax.get_xlim()
        YL = ax.get_ylim()
        lo = [XL[0], YL[0]]
        hi = [XL[1], YL[1]]
    
        if len(plot_props["xlim"]) > 0:
            ran = [plot_props["xlim"][0], plot_props["xlim"][1]]
        else:
            ran = [np.min(lo), np.max(hi)]
        ax.set_xlim(ran)
        ax.set_ylim(ran)
    
        # ---- Diagonal (ideal prediction) ----
        x_o = np.linspace(ran[0], ran[1], 1000)
        ax.plot(x_o, x_o, color=[0, 0, 0, 0.7], linewidth=1, linestyle='--', label='Ideal')
    
        # ---- Labels and ticks ----
        ax.set_xlabel("ground truth values")
        ax.set_ylabel("predicted values")

        if len(plot_props["xticks"]) > 0:
            ax.set_xticks(plot_props["xticks"])
        if len(plot_props["yticks"]) > 0:
            ax.set_yticks(plot_props["yticks"])
        if len(plot_props["ylim"]) > 0:
            ax.set_ylim(plot_props["ylim"])
        if len(plot_props["xlim"]) > 0:
            ax.set_xlim(plot_props["xlim"])
    
        # ---- Styling ----
        ax.set_aspect(1)
        self.style_plot(plt, ax)
        plt.show()
        
         # ---- Optional saving ----
        if tag != '':
            self.save_plot(fig, 'Preds-vs-Actual' + tag, folder)

   
          
        
    #     plt.show()
    def style_plot(self, plt, ax, plot_props=None,show=True):
        if plot_props is None:
            plot_props = {}
    
        # Helper function to get prop from plot_props or fallback to self.plt_prp
        def get_prop(key):
            return plot_props[key] if key in plot_props else self.plt_prp[key]
    
        # Line width for spines
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(get_prop('linewidth'))
    
        # Ticks
        if get_prop('top_ticks') == 'on':
            ax.tick_params(bottom=True, top=True)
            ax.tick_params(left=True, right=True)
    
        # Spines visibility
        ax.spines['right'].set_visible(get_prop('spine_right'))
        ax.spines['top'].set_visible(get_prop('spine_top'))
        ax.spines['left'].set_visible(get_prop('spine_left'))
        ax.spines['bottom'].set_visible(get_prop('spine_bottom'))
    
        # Tick width
        ax.tick_params(width=get_prop('linewidth'))
    
        # Legend
        if get_prop('legend') == 'on':
            plt.legend()
    
        # Axis font
        fontname = get_prop('fontname')
        ax.set_xlabel(ax.get_xlabel(), fontname=fontname)
        ax.set_ylabel(ax.get_ylabel(), fontname=fontname)
    
        # Tick label styling
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(fontname)
            tick.set_fontsize(get_prop('fontsize'))
            tick.set_fontweight(get_prop('fontweight'))
    
        # Axis label size & weight
        params = {
            'axes.labelsize': get_prop('fontsize_ax'),
            'axes.labelweight': get_prop('fontweight_ax')
        }
        
        plt.rcParams.update(params)
    
        # Grid
        plt.grid(get_prop('grid'))
    
        # Tick length
        tick_length = get_prop('tick_length')
        ax.tick_params(axis='both', which='major', length=tick_length)
        ax.tick_params(axis='both', which='minor', length=tick_length * 0.6)
    
        # Plot formatting from self.plot_format
        if hasattr(self, 'plot_format'):
            if 'xlim' not in plot_props:
                ax.set_xlim(self.plot_format["xlim"])
            else:
                ax.set_xlim(plot_props['xlim'])
    
            if 'ylim' not in plot_props:
                ax.set_ylim(self.plot_format["ylim"])
            else:
                ax.set_ylim(plot_props['ylim'])
    
            for label in ax.get_xticklabels():
                if 'xtick_format' in plot_props:
                    
                    label.set_rotation(plot_props["xtick_format"][0])
                    label.set_fontsize(plot_props["xtick_format"][1])
                else:
                    label.set_rotation(self.plot_format["xtick_format"][0])
                    label.set_fontsize(self.plot_format["xtick_format"][1])
    
            for label in ax.get_yticklabels():
                if 'ytick_format' in plot_props:
                    label.set_rotation(plot_props["ytick_format"][0])
                    label.set_fontsize(plot_props["ytick_format"][1])
                else:
                    label.set_rotation(self.plot_format["ytick_format"][0])
                    label.set_fontsize(self.plot_format["ytick_format"][1])
        
        
        for label in ax.get_xticklabels():

            label.set_rotation(get_prop("xtick_format")[0])
            label.set_fontsize(get_prop("xtick_format")[1])
            label.set_ha(get_prop("xtick_format")[2])   # or 'center', 'left'
            label.set_va(get_prop("xtick_format")[3])     # or 'center', 'bottom'
                
        for label in ax.get_yticklabels():
            label.set_rotation(get_prop("ytick_format")[0])
            label.set_fontsize(get_prop("ytick_format")[1])
            label.set_ha(get_prop("ytick_format")[2])   # or 'center', 'left'
            label.set_va(get_prop("ytick_format")[3])     # or 'center', 'bottom'
       
        
      
        # Additional settings that are only triggered if keys are explicitly present
        if 'y_axis_visible' in plot_props:
            ax.yaxis.set_visible(plot_props['y_axis_visible'])
            ax.spines['left'].set_visible(plot_props['y_axis_visible'])
    
        if 'x_axis_visible' in plot_props:
            ax.xaxis.set_visible(plot_props['x_axis_visible'])
            ax.spines['bottom'].set_visible(plot_props['x_axis_visible'])
    
        if 'x_scale' in plot_props:
            ax.set_xscale(plot_props['x_scale'])
            print('ADSfalfjlasdfj##############################################')
    
        if 'y_scale' in plot_props:
            ax.set_yscale(plot_props['y_scale'])
        if show:
            plt.show()
        
        
    def visualize_eigenpostures(self,num,paw_style=None,
                                tag='',folder=''):
 
        components = self.pca.components_
        raw = components
        std = np.tile(self.z_std, (raw.shape[0], 1))
        ms =  np.tile(self.z_means, (raw.shape[0], 1))
        raw = (raw*std+ms)
        pts = raw.reshape((raw.shape[0],self.num_keypoints,2))
        self.visualize_paws(pts, num,paw_style=paw_style,tag=tag,folder=folder)
    
    def compose_paws(self,sel_components,paw_style=None,
                                tag='',folder='',scaling=None,names=None):
        #if components == None:
        components = self.pca.components_[sel_components,:]
        
        
        group_centers = self.pc_group_centers[sel_components,:]
        sgc = group_centers.shape
        for i in range(group_centers.shape[1]):
            raw = np.sum(components*group_centers[:,i].reshape((sgc[0],1)),axis=0)
            raw = raw*self.z_std+self.z_means
            pts = raw.reshape((self.num_keypoints,2))
            if not isinstance(scaling, type(None)):
                hor_pts = pts[[0,scaling[2]],:]
   
                xl = np.sqrt(np.sum(np.diff(hor_pts,axis=0)**2,axis=1))[0]
                ver_pts = pts[[1,scaling[3]],:] #THIS IS HARDCODED.........
                yl = np.sqrt(np.sum(np.diff(ver_pts,axis=0)**2,axis=1))[0]
                x_fac = scaling[0]/xl
                y_fac = scaling[1]/yl
               
                pts[:,0] = pts[:,0]*x_fac
                pts[:,1] = pts[:,1]*y_fac
                
                hor_pts = pts[[1,scaling[2]],:]
                
                xl = np.sqrt(np.sum(np.diff(hor_pts,axis=0)**2,axis=1))[0]
               
            pts = raw.reshape((1,self.num_keypoints,2))
            if isinstance(names,type(None)):
                nu_tag = tag + '-ArcheType-' + str(i)
            else:
                nu_tag = tag +'-' + names[i]
            self.visualize_paws(pts, 1,paw_style=paw_style,tag=nu_tag,
                                folder=folder)   
    
    def plot_single_prediction(self,idx,boxType='absolute',directory=None):
        if directory is None:
            directory = str(self.label_db.iloc[idx].image_dir)
        
        img = cv2.imread(directory + '/' +self.label_db.iloc[idx].image_name)
        img = img[:,:,::-1]
        pts = self.pts[idx,:,:]
        bxs = self.boxes[idx,:,:]
        line = self.label_db.iloc[idx]        
        cols = ['genotype', 'gender','side','image_name','treatment']
        line = line[cols]
        shp = img.shape
        m = np.max(img)
        blank = np.tile(m,(round(shp[0]*0.3),shp[1],shp[2]))
        print('blank shape',blank.shape,img.shape)
        img = np.vstack((img,blank))
  
        text = '\n'.join([f"{col}: {val}" for col, val in line.items()])
        
           
        fig, ax = plt.subplots(dpi = 600)
        ax.imshow(img)
        
        self.visualize_paws(pts, 1,ax_obj = [fig,ax])
        print(bxs.shape)
        
        if boxType == 'absolute':
        
            x = [bxs[0,0],bxs[0,2],bxs[0,2],bxs[0,0],bxs[0,0]]
            y = [bxs[0,1],bxs[0,1],bxs[0,3],bxs[0,3],bxs[0,1]]
        
        elif boxType == 'relative':
            x = [bxs[0,0],bxs[0,0]+bxs[0,2],bxs[0,0]+bxs[0,2],bxs[0,0],bxs[0,0]]
            y = [bxs[0,1],bxs[0,1],bxs[0,1]+bxs[0,3],bxs[0,1]+bxs[0,3],bxs[0,1]]
        ax.plot(x,y)
        
        
        ax.text(shp[1]/2, shp[0]*1.25,text, ha='center', fontsize=10)
        plt.axis('off')
        plt.show()
            
    def show_angle(self,pts,ang_num,alpha,paw_style=None,folder='',tag='',scaling=None):
      
        if not isinstance(scaling, type(None)):
            hor_pts = pts[[0,scaling[2]],:]
            xl = np.sqrt(np.sum(np.diff(hor_pts,axis=0)**2,axis=1))[0]
            ver_pts = pts[[1,scaling[3]],:] #THIS IS HARDCODED.........
            yl = np.sqrt(np.sum(np.diff(ver_pts,axis=0)**2,axis=1))[0]
            x_fac = scaling[0]/xl
            y_fac = scaling[1]/yl
            pts[:,0] = pts[:,0]*x_fac
            pts[:,1] = pts[:,1]*y_fac

        fig,ax = plt.subplots(dpi=600)
        ax_obj = [fig,ax]
        ax_obj = self.visualize_paws(pts,1,alpha=alpha,ax_obj=ax_obj,
                                     paw_style=paw_style)
        edges = self.ang_comps[ang_num]
       
        for i in edges:

            try:
                idx = self.connect_logic.index(i)
            except:
                if ang_num == 142:
                    i = [i[1],i[0]]
                    idx = self.connect_logic.index(i)
                else:
                    raise Exception("Edge notation is invalid")
            
            ax_obj[1].plot([pts[i[0],0],pts[i[1],0]],[pts[i[0],1],pts[i[1],1]],
                           color=self.colors[idx],
                           linewidth=6,zorder=1)
            
            ax_obj[1].scatter(pts[i[0],0],pts[i[0],1],
                       color=self.colors[idx],s=65,zorder=2,
                       edgecolors=[0.2,0.2,0.2,1])
        name = "angle_indicator-" + tag
        self.save_plot(ax_obj[0],name,folder)
        
        
    def visualize_paws(self,pts,num,paw_style=None,ax_obj=None,tag='',folder='',
                       title=False,alpha=1):
        if len(pts.shape) != 3 :
            pts = pts.reshape((int(1),pts.shape[0],pts.shape[1]))
        pts = pts[:,:,[0,1]]
        redefine = False
        if ax_obj == None:
            redefine = True
        else:
            num = 1
            print('axes provided: number of paws plotted is set to 1')

            
        for i in range(num):
            if redefine:
                ax_obj = plt.subplots(dpi=600)
            ax_obj[0].set_size_inches(6, 6)
            ax_obj[1].set_aspect('equal', adjustable='box')
            if title :
                title = self.label_db.iloc[i].image_name + ' ' + str(i) 
                ax_obj[1].set_title(title)
            print('-------------------------------')
            
            point_colors = []
            searchMat = np.asarray(self.connect_logic)[:,0]
            for j in range(pts[i,:,:].shape[0]):
                
                if j == 0:

                    point_colors.append(self.colors[len(self.colors)-1])
             
                else:
                    a = np.where(searchMat == j)
                    
                    #print('a is',a[0][0])
                   
                    if len(a[0]) > 0:
                        point_colors.append(self.colors[int(a[0][0])])
        
                    else:
                        point_colors.append(point_colors[len(point_colors)-1])

                        
            for idx in range(pts[i,:,:].shape[0]):
                #print(i)
                ax_obj[1].scatter(pts[i,idx,0],pts[i,idx,1],
                           color=point_colors[idx],s=65,zorder=2,
                           edgecolors=[0.2,0.2,0.2,1],alpha=alpha)
            
            for idx,j in enumerate(self.connect_logic):

                ax_obj[1].plot([pts[i,j[0],0],pts[i,j[1],0]],[pts[i,j[0],1],
                        pts[i,j[1],1]],color=self.colors[idx],
                        linewidth=5,zorder=1,alpha=alpha)
            
            if isinstance(paw_style, dict):
                XL = ax_obj[1].get_xlim()
                d = XL[1]-XL[0]
                XL = [XL[0]-d*paw_style['xmargin'],
                    XL[1]+d*paw_style['xmargin']]
                ax_obj[1].set_xlim(XL)
                
                YL = ax_obj[1].get_ylim()
                d = YL[1]-YL[0]
                YL = [YL[0]-d*paw_style['ymargin'],
                      YL[1]+d*paw_style['ymargin']]
                ax_obj[1].set_ylim(YL)
                ax_obj[1].set_aspect(paw_style['aspect'], adjustable='box')
                ax_obj[1].axis(paw_style['axes'])
            
            if ax_obj==None:
                plt.show()
            
            if tag != '':
                name = 'PAW-' + str(i) + tag
                print(name)
                self.save_plot(ax_obj,name,folder)
        return ax_obj
           
    
    
    def analyze_residuals(self,residuals, bins=30,folder='',tag=''):
        """
        Analyze residuals:
        1. Plot histogram of residuals with ±1σ and ±3σ lines.
        2. Compute influence of each residual on the mean and
           plot average influence per bin using same bin edges.
        3. Optionally save bins and results to a .npz file.

        Parameters
        ----------
        residuals : array-like
            List or numpy array of residuals.
        bins : int or sequence, optional
            Number of bins or explicit bin edges for histogram.
        savefile : str or None, optional
            If provided, saves results (bin_edges, counts, avg_influence_per_bin).
        """

        residuals = np.asarray(residuals)
        n = len(residuals)

        # ---- 1. Histogram data ----
        counts, bin_edges = np.histogram(residuals, bins=bins)
        mean = np.mean(residuals)
        sigma = np.std(residuals)

        # ---- 2. Influence ----
        influence = residuals / n
        bin_indices = np.digitize(residuals, bin_edges) - 1
        avg_influence_per_bin = [
            np.mean(influence[bin_indices == i]) if np.any(bin_indices == i) else 0
            for i in range(len(bin_edges)-1)
        ]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # ---- 3. Create figure with subplots ----
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                                       gridspec_kw={'height_ratios': [1, 1]},
                                       dpi = 600)

        # Top plot: Histogram
        ax1.hist(residuals, bins=bin_edges, alpha=0.6, edgecolor="none")
        # for s in [1, 3]:
        #     ax1.axvline(mean + s*sigma, color="red", linestyle="dotted")
        #     ax1.axvline(mean - s*sigma, color="red", linestyle="dotted")
        ax1.axvline(mean + 3*sigma, color="red", linestyle="dotted")
     
        #ax1.set_ylabel("Frequency")

        # Bottom plot: Influence
        ax2.bar(bin_centers, np.array(avg_influence_per_bin)*counts,
                width=np.diff(bin_edges), alpha=0.6,color="lightcoral",
                edgecolor="none")
        # for s in [1, 3]:
        #     ax2.axvline(mean + s*sigma, color="red", linestyle="dotted")
        #     ax2.axvline(mean - s*sigma, color="red", linestyle="dotted")
        ax2.axvline(mean + 3*sigma, color="red", linestyle="dotted")
        #ax2.set_ylabel("Average influence")
        
        # Move x-axis to top and invert y-axis
        ax2.xaxis.set_ticks_position("top")
        ax2.xaxis.set_label_position("top")
        ax2.invert_yaxis()

        # Same x limits: from 0 to max residual bin edge
        ax1.set_xlim(0, bin_edges[-1])
        ax2.set_xlim(0, bin_edges[-1])

        plt.tight_layout()
        self.style_plot(plt, ax1,show=False)
        self.style_plot(plt, ax2)
        self.save_plot(fig,tag+'_residual_diagnostics',folder)
    
    
    def find_colors(self,u_groups,groups,base_colors):
        colors = []
        for i in groups:
            print(i,int(np.where(u_groups==i)[0]))
            colors.append(base_colors[int(np.where(u_groups==i)[0]),:])
        return np.asarray(colors) 
                
        
           
    
    
    def volcano_plot(self,df, delta_col, pvalue_col, delta_cutoff, pvalue_cutoff,
                     tag='',folder='',save_stats=False,show_plot=True,figsize=[6,6]):
        
        df = copy.copy(df)
        # Calculate -log10(p-value) for the y-axis
        df['-log10(pValue)'] = -np.log10(df[pvalue_col])
    
        # Define colors for points based on the cut-off values
        conditions = (abs(df[delta_col]) >= delta_cutoff) & (df[pvalue_col] <= pvalue_cutoff)
        colors = np.where(conditions, 'red', 'gray')
        if show_plot:
            fig,ax = plt.subplots(figsize=figsize,dpi=600)
            # Plot the data
            
            ax.scatter(df[delta_col], df['-log10(pValue)'],marker='^', c=colors, alpha=0.5)
            
            # Add dotted cut-off lines for delta and p-value
            ax.axvline(x=delta_cutoff, color='black', linestyle='--')
            ax.axvline(x=-delta_cutoff, color='black', linestyle='--')
            ax.axhline(y=-np.log10(pvalue_cutoff), color='black', linestyle='--')
            
            # Label the axes

            plt.xlabel('effect size (delta/σ)', fontdict={'fontsize': self.plt_prp["fontsize"], 'fontname': 'DejaVu Math TeX Gyre'})
            plt.ylabel('-log10(p-value)', fontdict={'fontsize': self.plt_prp["fontsize"], 'fontname': 'DejaVu Math TeX Gyre'})
            
            #self.style_plot(plt,ax)
           
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(self.plt_prp['linewidth'])
            # increase tick width
            ax.tick_params(width=self.plt_prp['linewidth'])
           
            # Generate values for the logistic curve
                   
            params = {'axes.labelsize': self.plt_prp['fontsize_ax'],
                      'axes.labelweight': self.plt_prp['fontweight_ax']}
            plt.rcParams.update(params)
            plt.grid(self.plt_prp['grid'])
            ax.tick_params(axis='both', which='major', length=self.plt_prp['tick_length'])
            ax.tick_params(axis='both', which='minor', length=self.plt_prp['tick_length']*0.6)
            if self.plt_prp['top_ticks'] == 'on':
                ax.tick_params(bottom=True,top=True)
                ax.tick_params(left=True,right=True)
            ax.spines['right'].set_visible(self.plt_prp['spine_right'])
            ax.spines['top'].set_visible(self.plt_prp['spine_top'])
            ax.spines['left'].set_visible(self.plt_prp['spine_left'])
            ax.spines['bottom'].set_visible(self.plt_prp['spine_bottom'])
            for tick in ax.get_xticklabels():
                tick.set_fontname(self.plt_prp['fontname'])
                tick.set_fontsize(self.plt_prp['fontsize'])
                tick.set_fontweight(self.plt_prp['fontweight'])
            for tick in ax.get_yticklabels():
                tick.set_fontname(self.plt_prp['fontname'])
                tick.set_fontsize(self.plt_prp['fontsize'])
                tick.set_fontweight(self.plt_prp['fontweight'])
            
            # Show the plot
            plt.show()
        
        
        
            self.save_plot(fig,'volcano_plot_'+tag, folder)
        
        # Return the filtered DataFrame where conditions meet the cut-off values
        filtered_df = df[conditions].copy()
        if save_stats:
            s_path = self.plot_path + '/' + folder +'/' + 'thresholded_satistics' + tag + '.csv'    
            filtered_df.to_csv(s_path, index=False)
        
        key_cols = [
            'comparison_number', 'angle_number', 'group1', 'group2',
            'mean1', 'mean2', 'delta', 'relative_delta',
            'SD1', 'SD2', 'SD12', 'n1', 'n2'
        ]
        
        # create an indicator of rows in df_large that are also in df_small
        mask = df[key_cols].merge(
            filtered_df[key_cols].drop_duplicates(),
            on=key_cols,
            how='left',
            indicator=True
        )['_merge'] == 'both'
        
        # set qVal and pVal = 0.051 where not in df_small
        df.loc[~mask, ['qVal', 'pVal']] = 0.051
        df.loc[~mask, ['indicator']] = 'ns.'        
        
        
        return filtered_df,df
    
    
    def map_significance_on_paw(self,df,paw_num,tag='',folder='',
                                color_map="RdBu",display='significant_angles',caxis=None):
        # check the statistics
        df = df[df['indicator'] != 'ns.']
        
        if display == 'significant_angles':
            counts = df['angle_number'].value_counts()
        
        elif display == 'summed_effect':
            df["abs_delta"] = df["relative_delta"].abs()
            df_sum = df.groupby("angle_number")["abs_delta"].sum().reset_index()
            counts = df_sum.set_index("angle_number")["abs_delta"]
        
        elif display == 'summed_angles':
            df["abs_delta"] = df["delta"].abs()
            df_sum = df.groupby("angle_number")["abs_delta"].sum().reset_index()
            counts = df_sum.set_index("angle_number")["abs_delta"]
        else: 
            raise Exception('No method specified. Please use significant_angles or summed_angles')
        
        all_edges = np.asarray(self.ang_comps)[counts.index.to_numpy(),:,:]
        edge_score = np.zeros((len(self.connect_logic),1))
        
        for idx,i in enumerate(self.connect_logic):
            ind1 = np.where(np.all(all_edges[:,0] == i, axis=1))[0]
            ind2 = np.where(np.all(all_edges[:,1] == i, axis=1))[0]
            edge_score[idx] = np.sum(counts.values[ind1]) + np.sum(counts.values[ind2])
            print(edge_score[idx])
        
        # this is for the color score 
        if caxis is None:
            norm_score = (edge_score - np.min(edge_score))/(np.max(edge_score)-np.min(edge_score))      
        else:
            norm_score = (edge_score - caxis[0])/(caxis[1]-caxis[0])  
        
        norm_score = norm_score.reshape((norm_score.shape[0]))
        
        fig,ax = plt.subplots(dpi=600)
        ax.set_aspect('equal', adjustable='box')  
        
        pts = self.pts_2_pca(self.pts,nth_point=6,
                             re_zero_type='mid_line',mirror_type='mid_line',
                             flatten=False)
        
        paw = pts[paw_num,:,0:2]
        paw[:,0] = paw[:,0]-np.min(paw[:,0])
        paw[:,1] = paw[:,1]-np.min(paw[:,1])        
    

       
        #gradient = np.linspace(0.2, 0.85, 100)
        cmap = plt.get_cmap(color_map)  # Example green-to-red colormap
        color_gradient = np.flipud(cmap(np.linspace(0, 1, 100)))
        

 
        c_idx = np.uint32(norm_score*99)
        #out-of-range handling:
        c_idx[c_idx>color_gradient.shape[0]-1] = color_gradient.shape[0]-1 
    
        for idx,j in enumerate(self.connect_logic):
   
            ax.plot([paw[j[0],0],paw[j[1],0]],[paw[j[0],1],paw[j[1],1]],
                    color=color_gradient[c_idx[idx],:],linewidth=6,zorder=1)
        for i in paw:
            ax.scatter(i[0],i[1],s=65, c='gray')
            
            
        ax.axis('off')
        XL = ax.get_xlim()
        YL = ax.get_ylim()
        ax.set_xlim(XL)
        ax.set_ylim(YL)
        l = (XL[1]-XL[0])/1000
        stX = l*20
        h = (YL[1]-YL[0])/1000
        stY = h*20
        ptX = stX
        for i in np.arange(0,100):
            ax.plot([ptX,ptX+l],[stY,stY],color=color_gradient[i,:],linewidth=8,zorder=1)
            #print(ptX+l-ptX,i)
            ptX += l
            
        if caxis is None:
            the_min = np.min(edge_score)
            the_max = np.max(edge_score)
        else:
           the_min = caxis[0]
           the_max = caxis[1]
        ax.text(stX - 75*l,stY,str(np.uint32(the_min)),fontsize=16,verticalalignment='center',rotation=270)
        ax.text(ptX +10*l ,stY,str(np.uint32(the_max)),fontsize=16,verticalalignment='center',rotation=270)
        
        self.save_plot(fig,'significance_map'+tag,folder)
   
           
    def paw_plot(self,angles,offset=0,err_ang=10,max_n=100,folder='',tag='',
                 headlines = None,fontsize=14,fs_indicator=24):
        ncorr = 0
        if angles.shape[0] == 1:
            angles = np.vstack((angles,angles))
            ncorr = 1
            
        # check the number of paws to plot
        print('angles are',angles.shape)
        
        if hasattr(self, 'paw_plot_settings'):
            offset = self.paw_plot_settings['offset']
            err_ang = self.paw_plot_settings['err_ang']
            max_n = self.paw_plot_settings['max_n']

        
        if angles.shape[0] == 1:
            print('')
            angs = angles[:,0,self.interesting_idcs]
            var = np.full(angs.shape,0)
        else: 
            angs = angles[:,0,self.interesting_idcs]
            angs_rad = np.deg2rad(angs)
            var = np.rad2deg(scipy.stats.circstd(angs_rad,axis=0))
            angs = np.rad2deg(scipy.stats.circmean(np.deg2rad(angs),axis=0))
            
            offset = np.rad2deg(scipy.stats.circmean(np.deg2rad(angs),axis=0))
            #opening_angle = np.rad2deg(scipy.stats.circmean(np.deg2rad(angles[:,:,81]),axis=0))
        
    
        # paw angle metrics----------------------------------------------------

        start_angles = [(90+offset+angs[1]) %360,
                        (90+offset) %360,
                        (90+offset-angs[2]) % 360,
                        (90+offset-angs[2]-angs[3]) % 360]
        
        end_angles = [(90+offset+angs[0]+angs[1]) % 360,
                        (90+offset+angs[1]) % 360,
                        (90+offset) % 360,
                        (90+offset-angs[2]) % 360]
        
        deltas = [];
        for idx,i in enumerate(start_angles):
            deltas.append(end_angles[idx]-i)
        
        
        
        
        radii = [2, 3,3,3]
        center_x = 0
        center_y = 0
        colors = ['rosybrown', 'rosybrown','rosybrown','rosybrown']
        #colors_err = ['salmon', 'salmon','salmon','salmon']
        colors_err = ['mistyrose', 'mistyrose','mistyrose','mistyrose']  
        
        
        #toe metrics-----------------------------------------------------------
        
        line_angles = [90+offset+angs[0]+angs[1],
                       90+offset+angs[1],
                       90+offset,
                       90+offset-angs[2],
                       90+offset-angs[2]-angs[3]] # in degrees
    
        line_starting_points = [(0, 0), (0, 0), (0, 0),(0,0),(0,0)]
        
        line_lengths = []
        err_leng_degree = 2*radii[3]*math.pi/360
        
        line_segment_percentages = []
        #[[2/3*100,1/6*100,1/6*100 ], [75, 12.5, 12.5], [75, 12.5, 12.5],[75, 12.5, 12.5],[75, 12.5, 12.5]]
        err_add = err_leng_degree*err_ang
        radii_2 = [radii[0],radii[1],radii[2],radii[3],radii[3]]
        
        for idx,i in enumerate(radii_2):
            tot_l = i+err_add
            line_lengths.append(tot_l)
            print('percents: ', [i/tot_l*100,err_add/2/tot_l*100,err_add/2/tot_l*100,])
            line_segment_percentages.append([i/tot_l*100,err_add/2/tot_l*100,err_add/2/tot_l*100,])

        
        line_segment_colors = [['saddlebrown', 'darkred','red'], 
                               ['saddlebrown', 'darkred','red'],
                               ['saddlebrown', 'darkred','red'],
                               ['saddlebrown', 'darkred','red'],
                               ['saddlebrown', 'darkred','red']]
        
        line_widths = [[4,2,1],
                       [4,2,1],
                       [4,2,1],
                       [4,2,1],
                       [4,2,1]]
        

        #plot 
        #print('')
        radii_err = np.asarray(radii) + var*err_leng_degree;
        
        fig, ax = plt.subplots(dpi=600,figsize=(6,6))
        ax.axis('equal')
        self.draw_circle_segments(ax,start_angles, end_angles, radii_err, center_x, center_y, colors_err)
        self.draw_circle_segments(ax,start_angles, end_angles, radii, center_x, center_y, colors)
        attPs,endpoints = self.draw_lines(ax,line_angles, line_starting_points, line_lengths, line_widths, line_segment_percentages, line_segment_colors)
        ax.plot([0, 0], [.1, -1], linewidth=10, solid_capstyle='round',color='saddlebrown',alpha=.5)
        n_leng = (angles.shape[0]-ncorr)/max_n
        ax.plot([0, 0], [.1, .1-n_leng], linewidth=10, solid_capstyle='round',color='saddlebrown')
        ax.plot([0, 0], [.1, .1-n_leng], linewidth=5, solid_capstyle='round',color='rosybrown')
        
        ax.plot([-.1, .15], [-.5, -.5], linewidth=1, solid_capstyle='round',color=[.3,0,0,0.5],alpha=.6)
        ax.plot([-.1, .2], [-1, -1], linewidth=1, solid_capstyle='round',color=[.3,0,0,0.5],alpha=.6)
        half_n = int(max_n/2)
        ax.text(0.25, -0.5, f'{half_n}', rotation=0, ha='left', va='center',fontsize=fontsize)
        ax.text(0.25, -1, f'{max_n}', rotation=0, ha='left', va='center',fontsize=fontsize)
        degree_r = line_angles[-1]-90
        #print('attPs',attPs)
        attP = attPs[1];
        attP2 = attPs[2];
        ax.text(attP[0]+.2,attP[1]+.2, f'{err_ang} °', rotation=degree_r, ha='left', va='top',fontsize=fontsize)
        ax.text(attP2[0]+.2,attP2[1]+.2, f'{err_ang/2} °', rotation=degree_r, ha='left', va='top',fontsize=fontsize)
        
        #Drawing paw indicators
        if not isinstance(headlines,type(None)): 
            print('in headlines')
            names = ['i','ii','iii','iv','v']
            heights = [4,4.3,4.3,4.3,4]
            for idx,i in enumerate(headlines):
                ax.text(endpoints[idx],heights[idx], names[idx], fontsize=fs_indicator, color=i)
        
        plt.axis('off')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
        self.save_plot(fig,'PAW_PLOT'+tag,folder)
        
        
    def draw_circle_segments(self,ax,start_angles, end_angles, radii, center_x, center_y, colors=('blue', 'green')):
        # Check if the number of start and end angles match
        if len(start_angles) != len(end_angles) or len(start_angles) != len(radii):
            raise ValueError("The number of start angles, end angles, and radii must be the same.")

        # Generate angles for the circle segment
        #angles = np.linspace(0, 360, 100)

        for start_angle, end_angle, radius, color in zip(start_angles, end_angles, radii, colors):
           
            # Calculate x and y coordinates for the circle segment
            x = np.append([center_x], center_x + radius * np.cos(np.radians(np.linspace(start_angle, end_angle, 100))))
            y = np.append([center_y], center_y + radius * np.sin(np.radians(np.linspace(start_angle, end_angle, 100))))

            # Plot the circle segment as a colored patch
            plt.fill(x, y, color)

        # Plot center of the circle
        ax.plot(center_x, center_y, 'ro')  # 'ro' means red circle marker

        # Set aspect ratio to equal to maintain circular shape
        ax.axis('equal')

    
    def draw_lines(self,ax,angles, starting_points, lengths, line_widths=None, segment_percentages=None, segment_colors=None):
        # Check if the number of angles, starting points, and lengths match
        if len(angles) != len(starting_points) or len(angles) != len(lengths):
            raise ValueError("The number of angles, starting points, and lengths must be the same.")

        # Initialize a figure
        
        counter = 0
        endpoints =[]
        # Loop through each line
        for angle, start, length, seg_percents, seg_colors,line_width in zip(angles, starting_points, lengths, segment_percentages, segment_colors,line_widths):
            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Calculate endpoint of the line
            end_x = start[0] + length * np.cos(angle_rad)
            end_y = start[1] + length * np.sin(angle_rad)
            attPts = []
            # Check if segment percentages are provided
            if seg_percents:
                # Calculate segment lengths based on percentages
                seg_lengths = [length * perc / 100 for perc in seg_percents]
         
                # Calculate cumulative lengths for segments
                cumulative_lengths = np.cumsum(seg_lengths)
              
                # Initialize start point of segment
                seg_start = np.array(start)
                #print('------------------------------------------------------')
                # Loop through each segment
                
                cumulative_lengths = np.flip(cumulative_lengths)
                seg_colors.reverse()
                line_width.reverse()
                s_counter = 0
                for seg_length, seg_color,lw in zip(cumulative_lengths, seg_colors,line_width):
                    
                    # Calculate endpoint of the segment
                    seg_end = seg_start + seg_length * np.array([np.cos(angle_rad), np.sin(angle_rad)])
                    #print('start',seg_start,'end',seg_end,'seg_length',seg_length)
                    # Plot the segment with specified color
                    if counter == 0:
                        zrd = 2
                    else:
                        zrd = 1
                    ax.plot([seg_start[0], seg_end[0]], [seg_start[1], seg_end[1]], color=seg_color, linewidth=lw, solid_capstyle='round',zorder=zrd)
                    attPts.append([seg_end[0],seg_end[1]])
                    if s_counter == 0:
                        endpoints.append(seg_end[0])
                    s_counter += 1        
                    
                    # Update start point for the next segment
                    #seg_start = seg_end
                
            else:
                # Plot the line with thicker width and rounded ends
                ax.plot([start[0], end_x], [start[1], end_y], linewidth=3, solid_capstyle='round')
            counter += 1
            
        return attPts, endpoints
        # Add labels and title
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    def sort_left_right_data(self,X,df,timepoints):
        consistent = X.shape[0]  == len(df) == timepoints.shape[0]
        if not consistent:
            raise Exception("input data is not consistent")
        animals = np.unique(df.animal_id)
        tps = np.unique(timepoints)
        X_left = np.tile(np.nan,(animals.size*tps.size,X.shape[1]))
        X_right = np.tile(np.nan,(animals.size*tps.size,X.shape[1]))
        counter = 0
        single_counter = 0
        
      
        for i in animals:
            idx = df.animal_id==i
            idxL = np.logical_and.reduce([df.side == 'left', idx])
            idxR = np.logical_and.reduce([df.side == 'right', idx])
            for t in tps: 
                idxt = np.logical_and.reduce([idx, timepoints == t])
                idxLt = np.logical_and.reduce([idxL,  timepoints == t])
                idxRt = np.logical_and.reduce([idxR,  timepoints == t])
                
                if np.sum(idxLt) == 1 and np.sum(idxRt) == 1 and np.sum(idxt) == 2:

                    X_left[single_counter,:] = X[idxLt,:]
                    X_right[single_counter,:] = X[idxRt,:]
                    counter += 2
                    single_counter += 1

        killDx = np.sum(np.isnan((X_right)),axis=1) == X_right.shape[1]
        X_right = np.delete(X_right,killDx,axis=0)
        X_left = np.delete(X_left,killDx,axis=0)
        return X_right,X_left
  
    def fuse_dataObjs(self,objs):
        
        if 'dataset_id' in self.label_db:
            last_id = np.max(self.label_db.dataset_id)
        
        else:
            if len(self.label_db) == 0:
            
                last_id = 0
            else:
                col = np.tile(0,(len(self.label_db)))
                self.label_db["dataset_id"] = col
                last_id = 1
        
        for i in objs:
            self.pts = np.vstack((self.pts,i.pts))
            self.angles = np.vstack((self.angles,i.angles))
            self.boxes = np.vstack((self.boxes,i.boxes))
            self.stat_vector = np.vstack((self.stat_vector,i.stat_vector))
            self.centers = np.vstack((self.centers,i.centers))
            self.p_dists = np.vstack((self.p_dists,i.p_dists))
            
            col = np.tile(last_id,(len(i.label_db)))
            if not 'dataset_id' in i.columns:
                i.label_db["dataset_id"] = col
            
            self.concat_labels(i.label_db)
            
            
            last_id += 1
            
        self.consistency_check()
                  
    def export_training_instances(self,indices,directory):
        
        occurrences_dict = {}
        image_list = np.unique(self.label_db.iloc[indices].image_name)
    # Iterate over the list of strings
        for image in image_list:
            # Find the indices where the string occurs in the specified column
            indices = self.label_db[self.label_db['image_name'] == image].index.tolist()
            occurrences_dict[image] = indices
        
      # some loop goes here that iterates over the unique images and their instances. 
        
        for im in occurrences_dict:
            idcs = occurrences_dict[im]
           
            counter = 0
            dict_out = []
            pts_out = np.zeros((len(idcs),self.pts.shape[1],3))
            bbox_out = np.zeros((len(idcs),4))
            print(len(idcs),pts_out.shape)
            for i in idcs:
                new_row = {'genotype': self.label_db.genotype[i],
                           'gender': self.label_db.gender[i],
                           'side': self.label_db.side[i],
                           'treatment': self.label_db.treatment[i],
                           'paw_posture':self.label_db.paw_posture[i],
                           'pain_status':self.label_db.pain_status[i],
                           'useful':self.label_db.useful[i],
                           'remark':self.label_db.remark[i],
                           'animal_ID':self.label_db.animal_id[i],
                           'image_id': self.label_db.image_name[i],
                           'paw_number':counter,} 
                
                # Append the dictionary to the DataFrame
                print('pts',pts_out[counter,:,0:2],'shape',pts_out[counter,:,0:2].shape)
                
                pts_out[counter,:,0:2] = self.pts[i,:,0:2]
                bbox_out[counter,:] = self.boxes[i,:]
               
                #preparing meta data for export
                new_row['visibility'] = np.ones((15,1),'uint32')
                new_row['truncated'] = 0
                fname = directory + '/' + self.label_db.image_name[i]
                img = cv2.imread(fname)
                
                new_row['height'] = img.shape[0]
                new_row['width'] = img.shape[1]
                dict_out.append(new_row)
                counter += 1
                
            self.save_to_mat(pts_out, bbox_out, dict_out, fname)
            

    def save_to_mat(self,pts, bxs, data_dict, image_name):
        """
        Saves Python objects into a .mat file in a specified cell array format.

        Parameters:
            pts (np.ndarray): Array to be placed in cell (1,2)
            bxs (np.ndarray): Array to be placed in cell (2,2)
            data_dict (dict): Dictionary containing 'genotype', 'paw_posture', and 'side'
            filename (str): Name of the output .mat file
        """
        
        filename = image_name[0:-4] + '.mat'
        cell_array = [["points", '', '', '', '', '', ''],  # 1
                      ["rois", '', '', '', '', '', ''],  # 2
                      ["visability", '', '', '', '', '', ''],  # 3
                      ["truncated", '', '', '', '', '', ''],  # 4
                      ["height", '', '', '', '', '', ''],  # 5
                      ["width", '', '', '', '', '', ''],  # 6
                      ["genotype", '', '', '', '', '', ''],  # 7
                      ["gender", '', '', '', '', '', ''],  # 8
                      ["side", '', '', '', '', '', ''],  # 9
                      ["treatment", '', '', '', '', '', ''],  # 10
                      ["paw_posture", '', '', '', '', '', ''],  # 11
                      ["pain_status", '', '', '', '', '', ''],  # 12
                      ["useful", '', '', '', '', '', ''],  # 13
                      ["remark", '', '', '', '', '', ''],  # 14
                      ["animal_ID", '','', '', '', '', '']]  # 15
        
        for i in range(len(pts)):
        # Create an empty cell array (MATLAB cell array is represented as a list of lists in Python)


        
        
            # Populate the cell array based on given coordinates
            cell_array[0][i+1] = pts[i]        # (1,2) -> cell_array[0][1]
            cell_array[1][i+1] = bxs[i]        # (2,2) -> cell_array[1][1]
            cell_array[2][i+1] = data_dict[i]["visibility"]    # (3,2) -> cell_array[2][1]
            cell_array[3][i+1] = data_dict[i]["truncated"] # (4,2) -> cell_array[3][1]
            cell_array[4][i+1] = data_dict[i]["height"]        # (5,2) -> cell_array[4][1]
            cell_array[5][i+1] = data_dict[i]["width"]
            cell_array[6][i+1] = data_dict[i]["genotype"]
            cell_array[7][i+1] = data_dict[i]["gender"]
            cell_array[8][i+1] = data_dict[i]["side"]
            cell_array[9][i+1] = data_dict[i]["treatment"]
            cell_array[10][i+1] = data_dict[i]["paw_posture"]
            cell_array[11][i+1] = data_dict[i]["pain_status"]
            cell_array[12][i+1] = data_dict[i]["useful"]
            cell_array[13][i+1] = data_dict[i]["remark"]
            cell_array[14][i+1] = data_dict[i]["animal_ID"]

            
            # Convert the cell array to a format that can be saved in a .mat file
        
        #delete empty fields
        print('pts.shape',pts.shape,len(pts))
        print('cell_arrau',len(cell_array),len(cell_array[0]))
        for j in cell_array:
            for i in np.arange(len(pts)+1,7)[::-1]:
                j.pop(i)
           
        
        
        mat_dict = {'varList': cell_array}

        # Save to .mat file
        scipy.io.savemat(filename, mat_dict)
    

    def omit_gapping_data(self,data,labels):
        if data.shape[0] != len(labels):
          print('Attention the data and labels have unequal sizes')  
        if labels.dtype == 'float64':
            idcs = np.invert(np.isnan(labels))
        else:
            idcs = np.invert(labels=='nan')
        labels = labels[idcs]
        
        if len(data.shape)==3:
            data = data[idcs,:,:]
        else:
            idcs = np.asarray(idcs)
            data = data[idcs.reshape((idcs.shape[0],)),:]
        return data,labels
    
    def omit_gapping_data2(self,data,labels):
        if data.shape[0] != len(labels):
          print('Attention the data and labels have unequal sizes')  
        if labels.dtype == 'float64':
            idcs = np.invert(np.isnan(labels))
        else:
            idcs = np.invert(labels=='nan')
        labels = labels[idcs]
        
        if len(data.shape)==3:
            data = data[idcs,:,:]
        else:
            idcs = np.asarray(idcs)
            data = data[idcs.reshape((idcs.shape[0],)),:]
            
        return data,labels,idcs
    
    def concat_labels(self,fuse_obj):
        print(fuse_obj)
        if isinstance(fuse_obj, dict):
            fuse_obj = pd.DataFrame(data=fuse_obj,index=[0])
        self.label_db = pd.concat([self.label_db, fuse_obj], ignore_index=True, sort=False)
        # Reset the index
        self.label_db = self.label_db.reset_index(drop=True)

    def load_image(self,file_path=None):
        # Create a Tkinter root window and hide it
        root = tk.Tk()
        root.withdraw()
    
        # Open the file dialog to select an image file
        if file_path == None:
            file_path = askopenfilename(
                title="Select an image file",
                
                #filetypes=[("Image files","*.JPG;*.jpeg;*.png;*.bmp;*.tiff")]
            )
        
        if not file_path:
            print("No file selected.")
            return None
    
        # Load the image using OpenCV
        image = cv2.imread(file_path)
    
        if image is None:
            print("Failed to load the image.")
            return None
        return image
    
    
    def save_data_with_dialog(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        filename = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
       
        if filename:
            try:
                with open(filename, 'wb') as file:
                    pickle.dump((self.pts, self.boxes,self.centers,
                                 self.p_dists,self.stat_vector,self.label_db,
                                 self.angles), file)
                print(f"Data saved to {filename}")
            except Exception as e:
                print(f"An error occurred while saving data: {e}")
        else:
            print("Save operation cancelled")
  
    def save_data_zip(self,filename=None):
        if filename is None:
            root = tk.Tk()
            root.withdraw()  # Hide the main tkinter window
            
            filename = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Zip files", "*.zip")])
            root.destroy()
        else:
            filename = Path(filename)
            
        if filename:
            try:
                # Create a temporary directory to store files before zipping
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save numpy arrays as .mat
                    scipy.io.savemat(os.path.join(tmpdir, "pts.mat"), {"pts": self.pts})
                    scipy.io.savemat(os.path.join(tmpdir, "boxes.mat"), {"boxes": self.boxes})
                    scipy.io.savemat(os.path.join(tmpdir, "centers.mat"), {"centers": self.centers})
                    scipy.io.savemat(os.path.join(tmpdir, "p_dists.mat"), {"p_dists": self.p_dists})

                    if np.any(self.stat_vector == None):
                        self.stat_vector = np.nan
                    scipy.io.savemat(os.path.join(tmpdir, "stat_vector.mat"), {"stat_vector": self.stat_vector})
                    scipy.io.savemat(os.path.join(tmpdir, "angles.mat"), {"angles": self.angles})
    
                    # Save dataframe as csv
                    if isinstance(self.label_db, pd.DataFrame):
                        self.label_db.to_csv(os.path.join(tmpdir, "label_db.csv"), index=False)
                    else:
                        raise ValueError("label_db must be a pandas DataFrame")
    
                    # Create zip archive
                    base_name = os.path.splitext(filename)[0]
                    shutil.make_archive(base_name, 'zip', tmpdir)
                
                print(f"Data saved to {filename}")
            except Exception as e:
                print(f"An error occurred while saving data: {e}")
        else:
            print("Save operation cancelled")
            
    def default_plot_props(self):
        if hasattr(self,'plot_format'):
            del self.plot_format
        
        self.plt_prp = {'linewidth':2.5,
                        'fontweight':'normal',
                        'fontweight_ax':'bold',
                        'fontsize':22,
                        'fontsize_ax':24,
                        'fontname':'Abyssinica SIL',#'DejaVu Sans Mono',
                        'legend':'off',
                        'top_ticks':'on',
                        'spine_right':True,
                        'spine_top':True,
                        'spine_left':True,
                        'spine_bottom':True,
                        'size':(6,6),
                        'dpi':600,
                        'grid':False,
                        'tick_length':6,
                        'xtick_format':[0,22,'center','top'],
                        'ytick_format':[0,22,'right','center']}
            
    
    def reset_all_data(self):
        self.pts = np.empty((0,self.num_keypoints,3),dtype=float)
        self.boxes = np.empty((0,1,4),dtype=float)
        self.centers = np.empty((0,1,2),dtype=float)
        self.p_dists = np.empty((0,int((self.num_keypoints*self.num_keypoints-self.num_keypoints)/2)),dtype=float)
        
        #self.instance_identy = []
        self.stat_vector = None
        # updated by by collect_data
        self.label_db = pd.DataFrame(columns = ['genotype','gender','side','treatment','paw_posture','pain_status','image_name','paw_index','remark','animal_id','useful'])
        #updated by .all_angles()
        self.angles = np.empty((0,1,len(self.ang_comps)),dtype=float)
        print("All data deleted")
        
    
    def load_data_with_dialog(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        filename = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        content = []
        root.destroy()
        if filename:
            try:
                with open(filename, 'rb') as file:
                    content = pickle.load(file)
                print(f"Data loaded from {filename}")
                self.pts = content[0]
                self.boxes = content[1]
                self.centers = content[2]
                self.p_dists = content[3]
                self.stat_vector = content[4]
                self.label_db = content[5]
                self.angles = content[6]
            except Exception as e:
                print(f"An error occurred while loading data: {e}")
                return None, None
        else:
            print("Load operation cancelled")
        self.consistency_check()

    
    def load_data_zip(self,filename=None):
        if filename is None:
            root = tk.Tk()
            root.withdraw()  # Hide the main tkinter window
            
            filename = filedialog.askopenfilename(filetypes=[("Zip files", "*.zip")])
            root.destroy()
        else:
            filename = Path(filename)
        if filename:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Extract zip into temp directory
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
    
                    # Load MATLAB arrays
                    self.pts = scipy.io.loadmat(os.path.join(tmpdir, "pts.mat"))["pts"]
                    self.boxes = scipy.io.loadmat(os.path.join(tmpdir, "boxes.mat"))["boxes"]
                    self.centers = scipy.io.loadmat(os.path.join(tmpdir, "centers.mat"))["centers"]
                    self.p_dists = scipy.io.loadmat(os.path.join(tmpdir, "p_dists.mat"))["p_dists"]
                    self.stat_vector = scipy.io.loadmat(os.path.join(tmpdir, "stat_vector.mat"))["stat_vector"]
                    self.angles = scipy.io.loadmat(os.path.join(tmpdir, "angles.mat"))["angles"]
    
                    # Load dataframe
                    self.label_db = pd.read_csv(os.path.join(tmpdir, "label_db.csv"))
    
                print(f"Data loaded from {filename}")
            except Exception as e:
                print(f"An error occurred while loading data: {e}")
                return None, None
        else:
            print("Load operation cancelled")

        self.consistency_check()
            
    def load_from_matlab(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        filenames = filedialog.askopenfilename(filetypes=[("Pickle files", "*.mat")],multiple=True)
        print(filenames)
        for i in filenames:
            tf = scipy.io.loadmat(i)
            num_instances = len(tf['varList'][0])
            for k in range(1,num_instances):
                
                #print(tf['varList'][0][0],tf['varList'][1][0],tf['varList'][2][0],tf['varList'][3][0],tf['varList'][4][0],tf['varList'][5][0],tf['varList'][6][0])
                bx = tf['varList'][2][k];
                #print('num is:', k)
               # print('shapes are:',tf['varList'][0][k].shape,tf['varList'][2][k].shape)
                a = np.hstack((tf['varList'][0][k],bx.reshape((bx.shape[1],bx.shape[0]))))
                #print(self.pts.shape)
                #print(a.shape)
                b = tf['varList'][1][k]
                
                self.add_data(a.reshape((1,self.num_keypoints,3)),b.reshape((1,1,4)))
                
                new_row = {'genotype': tf['varList'][6][k],
                           'gender': tf['varList'][7][k],
                           'side': tf['varList'][8][k],
                           'treatment': tf['varList'][9][k],
                           'paw_posture':tf['varList'][10][k],
                           'pain_status':tf['varList'][11][k],
                           'useful':tf['varList'][12][k],
                           'remark':tf['varList'][13][k],
                           'animal_id':tf['varList'][14][k]}

                    # Append the dictionary to the DataFrame
                self.concat_labels(new_row)
        self.all_angles()
        self.update_pdist(normalize=True)
        
    def load_single_matlab(self,filename=None):
        root = tk.Tk()
        root.withdraw()  
        if filename == None:
            filename = filedialog.askopenfilename(filetypes=[("Pickle files", "*.mat")],multiple=False)
        tf = scipy.io.loadmat(filename)
        num_instances = len(tf['varList'][0])
        bxs = np.empty((num_instances-1,1,4))
        pts = np.empty((num_instances-1,self.num_keypoints,3)) 
        meta_data = [];

        for k in range(1,num_instances):
            print(tf['varList'][0][0],tf['varList'][1][0],tf['varList'][2][0],tf['varList'][3][0],tf['varList'][4][0],tf['varList'][5][0],tf['varList'][6][0])
            print('num is:', k)

            #bx = bx.reshape((bx.shape[1],bx.shape[0]))
            
            pt = tf['varList'][0][k]
     
            
            bx = tf['varList'][1][k]
            
            if pt.shape[1] == 2:
                pt = np.hstack((pt,np.ones((self.num_keypoints,1))))  
             

            dat = pt.reshape((1,self.num_keypoints,3))#self.num_keypoints
            bxs[k-1,:,:] = bx.reshape((1,1,4))
            pts[k-1,:,:] = dat
            new_row = {'genotype': tf['varList'][6][k],
                           'gender': tf['varList'][7][k],
                           'side': tf['varList'][8][k],
                           'treatment': tf['varList'][9][k],
                           'paw_posture':tf['varList'][10][k],
                           'pain_status':tf['varList'][11][k],
                           'useful':tf['varList'][12][k],
                           'remark':tf['varList'][13][k],
                           'animal_id':tf['varList'][14][k],
                           'visibility':tf['varList'][2][k],
                           'truncated':tf['varList'][3][k],
                           'height':tf['varList'][4][k],
                           'width':tf['varList'][5][k]}
            meta_data.append(new_row)
        return pts,bxs,meta_data           
 
               
       
    
    def load_data_from_df(self,df_path,image_path,reFrame=False,override=False):
        df = pd.read_csv(df_path)
        df_new = pd.DataFrame(columns=['useful'])
        #check if there are doubles in the image ID...
        added_indices = []
        added = []
        files = df.image_name
        unique_files = np.unique(files)

        i = 0
        dec = 1
        while i < len(unique_files):
            if dec == -1 and added[-2]:
                
                self.pts = np.delete(self.pts, added_indices[-1], axis=0)
                self.boxes = np.delete(self.boxes, added_indices[-1], axis=0)
                df_new = df_new.drop(added_indices[-1], axis=0)
                
                del added[-1]
                del added_indices[-1]
           
            added.append(False)
            file = unique_files[i]
            num_paws = np.count_nonzero(files == file)
            paw_indices = np.where(files == file)[0]
            # go through all the occurances of the same image
            
            all_pts = np.empty((num_paws,self.num_keypoints,3),dtype=float)
            all_bxs = np.empty((num_paws,1,4),dtype=float)
            meta_data = pd.DataFrame()
            to_add = []
            
            for sub_idx, df_idx in enumerate(paw_indices):
                    pts,bx,img,row,dec = self.fetch_single_instance_from_df(df,
                                                    df_idx,image_path,
                                                     override=override)
                    if dec == -1 or dec == -2:

                        break
                    elif dec == 1:
 
                        
                        
                        meta_data = pd.concat([meta_data, row], axis=0,
                                           ignore_index=True)
                        if hasattr(bx, 'tensor'):
                            bx = bx.tensor.numpy()
                        
                        all_pts[sub_idx,:,:] = pts
                        all_bxs[sub_idx,:,:] = bx
                        to_add.append(sub_idx + i)
                        
     
                    elif dec == 0:
                        print('No paw recognized in ',df.iloc[df_idx].image_name)
                        if 'useful' not in row.columns:
                            cf_new = pd.DataFrame(columns=['useful'])
                            row = pd.concat([row, cf_new], axis=0,
                                               ignore_index=True)
                        row.iloc[0].useful = ['no']                     
                        dec = 1
                        
            # saving the         
            if dec == 1:
                
                #saving the data 
                
                if len(all_bxs.shape) > 2:
                    #print('darn')
                    all_bxs = all_bxs.reshape((all_bxs.shape[0],all_bxs.shape[2]))

                self.add_data(all_pts, all_bxs)
                df_new = pd.concat([df_new, meta_data], axis=0,
                                   ignore_index=True)
                
                
                
                


                mat_name = image_path + '/' + file[0:-4] + '.mat'

                dict_out = self.prepare_dict(meta_data,img.shape)

                self.save_to_mat(all_pts, all_bxs, dict_out, mat_name)
                
                added_indices.append(to_add)
                added[i] = True
                
                
            #prepare the dictionary. 
            if dec == -2:
                break 
            
            i += dec 
            
            
        self.concat_labels(df_new)
        self.all_angles()
        self.consistency_check()        
    
    def repair_boxes_and_centers(self):
        max_idx = len(self.label_db)
        for i in range(0,max_idx):
           try:
               t = self.boxes[i,:,:]
           except:
              a = np.array([np.min(self.pts[i,:,0]),
                                             np.max(self.pts[i,:,1]),
                                             np.max(self.pts[i,:,0]),
                                             np.min(self.pts[i,:,1])])
              self.boxes = np.vstack((self.boxes,a.reshape((1,1,4))))
              
        for i in range(0,max_idx):
           try:
               t = self.centers[i,:,:]
           except:
               a = np.array([np.mean(self.pts[i,:,0]),
                                             np.mean(self.pts[i,:,1])])
               self.centers = np.vstack((self.centers,a.reshape((1,1,2))))
               
        self.consistency_check()
            
  
        
    def prepare_dict(self,df,img_size):
        def ensure_column_exists(df, column_name, default_value='not stated'):
            if column_name not in df.columns:
                df[column_name] = default_value
                #print(f"Column '{column_name}' added with default value: {default_value}")
            return df
        
        #print('row is', df)
        ensure_column_exists(df, "genotype")
        ensure_column_exists(df, "gender")
        ensure_column_exists(df, "side")
        ensure_column_exists(df, "paw_posture")
        ensure_column_exists(df, "pain_status")
        ensure_column_exists(df, "useful")
        ensure_column_exists(df, "remark")
        ensure_column_exists(df, "animal_id")
        ensure_column_exists(df, "image_name")
        dict_out = []

        for i in range(0,len(df)):
            new_row = {'genotype': df.genotype.iloc[i],
                       'gender': df.gender.iloc[i],
                       'side': df.side.iloc[i],
                       'treatment': df.treatment.iloc[i],
                       'paw_posture':df.paw_posture.iloc[i],
                       'pain_status':df.pain_status.iloc[i],
                       'useful':df.useful.iloc[i],
                       'remark':df.remark.iloc[i],
                       'animal_ID':df.animal_id.iloc[i],
                       'image_id': df.image_name.iloc[i],
                       
                       'paw_number':i,}
            new_row['visibility'] = np.ones((15,1),'uint32')
            new_row['truncated'] = 0
            new_row['height'] = img_size[0]
            new_row['width'] = img_size[1]
            dict_out.append(new_row)
            
            # Append the dictionary to the DataFrame
        return dict_out
            
      

    
    def save_plot(self,axObj,name,folder):
        s_path = self.plot_path + '/' + folder +'/'
        print(s_path)
        if not os.path.exists(s_path):
            os.mkdir(s_path)
            
        if self.session_id == None:
            fname = s_path + name + '.png'
            pickle_name = s_path + name + '.pickle'
        else:
            fname = s_path + self.session_id + '-' + name + '.png'
            pickle_name = s_path + self.session_id + '-' + name + '.pickle'
            
        if isinstance(axObj,tuple):
            axObj[0].savefig(fname, bbox_inches='tight')
            fig = axObj[0]
        else:
            axObj.savefig(fname, bbox_inches='tight')
            fig = axObj
        
        with open(pickle_name, 'wb') as f:
            pickle.dump(fig, f)
        
    def load_plot(self,name,folder):
        s_path = self.plot_path + '/' + folder +'/'
        print(s_path)
        if not os.path.exists(s_path):
            os.mkdir(s_path)
            
        if self.session_id == None:

            pickle_name = s_path + name + '.pickle'
        else:

            pickle_name = s_path + self.session_id + '-' + name + '.pickle'
        with open(pickle_name, 'rb') as f:
            fig = pickle.load(f)
        plt.figure(fig.number)
        plt.show()
        
        return fig,fig.axes        
    
